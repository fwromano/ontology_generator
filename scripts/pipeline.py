# pipeline.py
import os
import subprocess

from parse_manual import extract_text_from_pdf
from generate_ontology import generate_ontology_for_document
from visualize_ontology import visualize_ontology

def process_pdf_to_ontology_and_visualize(
    pdf_path, 
    output_ttl, 
    output_dot):
    """
    1) Extract text from PDF
    2) Generate an ontology TTL from that text via LLM
    3) Visualize the TTL (producing a .dot)
    """
    # ---- (1) Extract text from PDF ----
    extracted_text = extract_text_from_pdf(pdf_path)
    print(f"[INFO] Extracted {len(extracted_text)} characters from '{pdf_path}'")
    
    # ---- (2) Generate TTL file via LLM ----
    generate_ontology_for_document(extracted_text, output_ttl=output_ttl)
    print(f"[INFO] Turtle file saved to '{output_ttl}'")
    
    
    # ---- (3) Visualize (TTL -> DOT) ----
    visualize_ontology(turtle_file=output_ttl, output_dot=output_dot)
    print(f"[INFO] DOT file saved to '{output_dot}'")

def process_ttl_to_visualize(ttl_path, output_dot):
    """
    Visualize an existing TTL file
    """
    visualize_ontology(turtle_file=ttl_path, output_dot=output_dot)
    print(f"[INFO] DOT file saved to '{output_dot}'")


if __name__ == "__main__":
    # # 1) List all PDFs in data/ folder
    # data_folder = "data/primary_documents"
    # pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]
    
    # if not pdf_files:
    #     print(f"[ERROR] No PDF files found in '{data_folder}' folder.")
    #     exit(1)

    # print("Available PDF files:")
    # for idx, fname in enumerate(pdf_files, start=1):
    #     print(f"{idx}. {fname}")

    # # 2) Prompt user to pick a file
    # choice = input("\nEnter the number of the PDF you want to process: ")
    # try:
    #     choice_idx = int(choice) - 1
    #     if choice_idx < 0 or choice_idx >= len(pdf_files):
    #         raise ValueError
    # except ValueError:
    #     print("[ERROR] Invalid selection.")
    #     exit(1)

    # chosen_pdf = pdf_files[choice_idx]
    # pdf_path = os.path.join(data_folder, chosen_pdf)
    
    # # 3) Generate matching filenames
    # base_name = os.path.splitext(chosen_pdf)[0]  # removes ".pdf"
    # ttl_file = f"ontologies/{base_name}_ontology.ttl"
    # dot_file = f"ontologies/graphs/{base_name}_ontology.dot"

    # # 4) Run the pipeline
    # process_pdf_to_ontology_and_visualize(
    #     pdf_path=pdf_path,
    #     output_ttl=ttl_file,
    #     output_dot=dot_file    )
    
    # print("\n[DONE] Full pipeline executed.")
    # print(f"Generated files:\n  TTL: {ttl_file}\n  DOT: {dot_file}\n")

    ttl_path = "4omTTL.ttl"
    dot_file = "4omDOT.dot"
    process_ttl_to_visualize(ttl_path=ttl_path, output_dot=dot_file)
    print("\n[DONE] Visualized existing TTL file.")
    print(f"Generated DOT file: {dot_file}")
    