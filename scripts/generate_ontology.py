import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
from validation import clean_and_validate_ttl

load_dotenv()

#########################################
#  Setup your LLM API client
#########################################
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


#########################################
#   Step 0:  Text Splitting Utility
#########################################
def chunk_text(document: str, chunk_size=50000, overlap=4000) -> List[str]:
    """
    Splits `document` into overlapping chunks to handle large texts.
    Adjust chunk_size & overlap to your model’s token limit / desired context overlap.
    """
    chunks = []
    start = 0
    while start < len(document):
        end = min(start + chunk_size, len(document))
        chunk = document[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


#########################################
#   Step 1: Partial Extraction per Chunk
#########################################
def extract_partial_json(chunk: str) -> Dict:
    """
    Prompts the LLM to extract a structured JSON 'partial ontology' from a chunk.
    Example JSON schema:
    {
      "entities": [...],
      "relationships": [...],
      "properties": [...],
      "notes": ...
    }
    Adapt the schema to your domain needs.
    """

    prompt = f"""
    Instruction:

    Please read and analyze the following document that explains an OWL ontology. Your task is to convert the ontology's structure and components into a well-organized JSON format. The JSON should comprehensively capture the ontology's classes, properties (both object and data properties), individuals, and their interrelationships. Ensure that hierarchical relationships, property domains and ranges, and any annotations or descriptions are accurately represented.

    **Requirements for the JSON Structure:**

    1. **Classes:**
        - **Name:** The name of the class.
        - **Description:** A brief description or definition of the class.
        - **Superclasses:** List of parent classes (if any).
        - **Subclasses:** List of child classes (if any).
        - **Properties:** Properties associated with the class.

    2. **Properties:**
        - **Type:** Specify whether it's an "Object Property" or "Data Property."
        - **Domain:** The class to which the property belongs.
        - **Range:** The class or datatype that the property points to.
        - **Description:** A brief description of the property's purpose.
        - **Restrictions:** Any cardinality or value restrictions applied to the property.

    3. **Individuals:**
        - **Name:** The name of the individual.
        - **Class Membership:** The class or classes the individual belongs to.
        - **Property Values:** Key-value pairs representing property assignments.

    4. **Annotations:**
        - **Label:** Human-readable labels for classes, properties, or individuals.
        - **Comments:** Additional comments or notes providing more context.


    Document chunk:{chunk}
    """

    # having issue with this line. 
    #   prompt issue? 
    response = client.chat.completions.create(
        model="o3-mini-2025-01-31",
        messages=[
            {"role": "system", "content": "You are an expert in domain knowledge extraction."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    print(response)

    raw_json_str = response.choices[0].message.content.strip()

    cleaned_json = clean_and_validate_ttl(raw_json_str)
    print(cleaned_json)
    # Attempt to parse JSON
    try:
        partial_data = json.loads(cleaned_json)
    except json.JSONDecodeError:
        partial_data = {"entities": [], "relationships": [], "error": f"Invalid JSON returned: {cleaned_json[:200]}..."}
    # print(cleaned_json)
    return partial_data


#########################################
#   Step 1b: Merge Partial Extractions
#########################################
from typing import List, Dict, Set
from collections import defaultdict

def merge_partial_json(partials: List[Dict]) -> Dict:
    """
    Combines partial ontologies with conflict resolution and consistency checks.
    Features:
    - Deduplicates entities based on name
    - Merges entity descriptions from multiple sources
    - Validates relationship references
    - Handles conflicting property values
    - Maintains provenance information
    """
    
    # Helper functions for normalization
    def normalize_name(name: str) -> str:
        return name.strip().lower().replace(' ', '_')
    
    def create_relationship_key(rel: Dict) -> tuple:
        return (normalize_name(rel['source']), 
                normalize_name(rel['type']),
                normalize_name(rel['target']))

    # Storage structures with provenance tracking
    entities = defaultdict(dict)
    relationships = defaultdict(list)
    entity_provenance = defaultdict(set)
    relationship_provenance = defaultdict(set)
    
    # First pass: collect all entities and relationships
    for idx, partial in enumerate(partials):
        # Process entities
        for entity in partial.get('entities', []):
            name = normalize_name(entity['name'])
            
            # Record provenance
            entity_provenance[name].add(idx)
            
            # Merge properties with conflict resolution
            if name not in entities:
                entities[name] = entity
                entities[name]['sources'] = [idx]
            else:
                # Merge descriptions from different sources
                if 'description' in entity:
                    existing_desc = entities[name].get('description', '')
                    new_desc = entity['description'].strip()
                    if new_desc and new_desc not in existing_desc:
                        entities[name]['description'] = f"{existing_desc}\n{new_desc}".strip()
                
                # Merge other properties (last partial wins)
                for key in [k for k in entity if k not in ('name', 'description')]:
                    entities[name][key] = entity[key]
                
                entities[name]['sources'].append(idx)
        
        # Process relationships
        for rel in partial.get('relationships', []):
            if not all(key in rel for key in ['source', 'type', 'target']):
                continue  # Skip invalid relationships
            
            key = create_relationship_key(rel)
            relationship_provenance[key].add(idx)
            
            # Check if relationship already exists
            exists = False
            for existing in relationships[key]:
                if existing['properties'] == rel.get('properties', {}):
                    exists = True
                    break
            
            if not exists:
                relationships[key].append({
                    'source': normalize_name(rel['source']),
                    'target': normalize_name(rel['target']),
                    'type': rel['type'].strip(),
                    'properties': rel.get('properties', {}),
                    'sources': [idx]
                })
    
    # Second pass: validate relationships against entities
    valid_relationships = []
    for rel_list in relationships.values():
        for rel in rel_list:
            # Check if endpoints exist
            source_exists = normalize_name(rel['source']) in entities
            target_exists = normalize_name(rel['target']) in entities
            
            if source_exists and target_exists:
                valid_relationships.append(rel)
            else:
                # Optionally log missing entities here
                pass
    
    # Convert entities to list format
    final_entities = []
    for name, data in entities.items():
        entity = {
            'name': data['name'],
            'description': data.get('description', ''),
            'properties': data.get('properties', {}),
            'source_chunks': sorted(data['sources'])
        }
        final_entities.append(entity)
    
    # Add reverse relationships for validation
    relationship_types = set(r['type'] for r in valid_relationships)
    
    return {
        'entities': final_entities,
        'relationships': valid_relationships,
        'provenance': {
            'entity_sources': {k: sorted(v) for k, v in entity_provenance.items()},
            'relationship_sources': {k: sorted(v) for k, v in relationship_provenance.items()}
        },
        'stats': {
            'total_entities': len(final_entities),
            'total_relationships': len(valid_relationships),
            'unique_relationship_types': len(relationship_types)
        }
    }
#########################################
#   Step 2: Final Refinement to TTL
#########################################
def combine_json_into_ttl(merged_data: Dict) -> str:
    """
    Sends the aggregated JSON back to the LLM with instructions to produce a single, 
    coherent Turtle ontology. 
    """
    # Convert merged_data to a string
    merged_json_str = json.dumps(merged_data, indent=2)

    prompt = f"""
    We have a combined ontology in JSON format. Convert it into a coherent OWL/Turtle ontology.
    Include classes, relationships as OWL object properties, etc. 
    Return valid .ttl with no extra text.

    JSON data:
    {merged_json_str}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are an ontology expert familiar with JSON -> Turtle conversions."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    ttl_output = response.choices[0].message.content.strip()
    return ttl_output

def onepassllm(document_text):
    
    prompt = f"""
    Instruction:

    Please read and analyze the following document that explains an OWL ontology. Your task is to convert the ontology's structure and components into a valid, coherent OWL/Turtle ontology .ttl file.
    Include classes, relationships as OWL object properties, etc.
      The .ttl file should comprehensively capture the ontology's classes, properties (both object and data properties), individuals, and their interrelationships. 
      Ensure that hierarchical relationships, property domains and ranges, and any annotations or descriptions are accurately represented.

    **Requirements:**

    1. **Classes:**
        - **Name:** The name of the class.
        - **Description:** A brief description or definition of the class.
        - **Superclasses:** List of parent classes (if any).
        - **Subclasses:** List of child classes (if any).
        - **Properties:** Properties associated with the class.

    2. **Properties:**
        - **Type:** Specify whether it's an "Object Property" or "Data Property."
        - **Domain:** The class to which the property belongs.
        - **Range:** The class or datatype that the property points to.
        - **Description:** A brief description of the property's purpose.
        - **Restrictions:** Any cardinality or value restrictions applied to the property.

    3. **Individuals:**
        - **Name:** The name of the individual.
        - **Class Membership:** The class or classes the individual belongs to.
        - **Property Values:** Key-value pairs representing property assignments.

    4. **Annotations:**
        - **Label:** Human-readable labels for classes, properties, or individuals.
        - **Comments:** Additional comments or notes providing more context.


    Document chunk:{document_text}
    """

    # having issue with this line. 
    #   prompt issue? 
    response = client.chat.completions.create(
        model="o3-mini-2025-01-31",
        messages=[
            {"role": "system", "content": "You are an expert in domain knowledge extraction."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    print(response)



#########################################
#   The Two-Stage Main Function
#########################################
# New constants for output directories
PARTIALS_DIR = "data/partial_ontologies"
MERGED_PATH = "data/merged_ontology.json"

def generate_ontology_for_document(document_text: str, output_ttl="final_ontology.ttl"):
    # Create directories if they don't exist
    os.makedirs(PARTIALS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(output_ttl), exist_ok=True)

    # # --(1) Split doc--
    # chunks = chunk_text(document_text)

    # # --(2) Extract partial JSON per chunk--
    # partial_ontologies = []
    # for i, c in enumerate(chunks, start=1):
    #     print(f"[INFO] Processing chunk {i}/{len(chunks)} (length={len(c)} chars)")
    #     partial = extract_partial_json(c)
        
    #     # Save partial JSON to file
    #     partial_path = os.path.join(PARTIALS_DIR, f"partial_{i}.json")
    #     with open(partial_path, "w") as f:
    #         json.dump(partial, f, indent=2)
    #     print(f"Saved partial ontology to {partial_path}")
        
    #     partial_ontologies.append(partial)

    # # --(3) Merge partial JSONs--
    # merged_data = merge_partial_json(partial_ontologies)
    
    # # Save merged JSON
    # with open(MERGED_PATH, "w") as f:
    #     json.dump(merged_data, f, indent=2)
    # print(f"Saved merged JSON to {MERGED_PATH}")

    # print("[INFO] Merged partial JSON. Entities:", len(merged_data["entities"]), 
    #       "Relationships:", len(merged_data["relationships"]))

    # # --(4) Final LLM call to produce TTL--
    # final_ttl = combine_json_into_ttl(merged_data)


    final_ttl = onepassllm(document_text)
    # Clean or validate if you have a function for TTL
    final_ttl = clean_and_validate_ttl(final_ttl)

    # --(5) Save final TTL--
    with open(output_ttl, "w", encoding="utf-8") as f:
        f.write(final_ttl)
    print(f"[DONE] Wrote final ontology to {output_ttl}")

    return output_ttl
