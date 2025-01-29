import rdflib
import pydot
from rdflib.namespace import RDFS, RDF, OWL

def get_rdfs_label(graph, uri):
    """
    Returns the first rdfs:label for `uri` if present,
    otherwise returns the local name (URI fragment).
    """
    label = next(graph.objects(uri, RDFS.label), None)
    if label:
        return str(label)
    # fallback to the portion after '#' or last '/'
    return uri.split('#')[-1] or uri.rsplit('/', 1)[-1]

def get_rdfs_comment(graph, uri):
    """
    Returns the first rdfs:comment for `uri` if present, else ''.
    """
    cmt = next(graph.objects(uri, RDFS.comment), None)
    return str(cmt) if cmt else ""

def visualize_ontology(turtle_file, output_dot="ontology_graph.dot"):
    """
    Parses the .ttl file and produces a .dot graph with:
      - Classes (shape=box, labeled by rdfs:label)
      - Subclass-of edges (blue dotted)
      - ObjectProperty domain->range edges (gray, deduplicated)
      - Individuals (shape=ellipse, labeled by rdfs:label)
      - Instance-level ObjectProperty edges (green)
      - rdfs:comment displayed in node tooltips
      - Uses a strict graph to avoid duplicate edges
    """
    g = rdflib.Graph()
    g.parse(turtle_file, format="ttl")

    # Use strict=True so parallel edges get merged automatically
    dot = pydot.Dot(graph_type='digraph', rankdir='LR', fontsize='10', strict=True)

    def add_or_update_node(uri, shape, color, fillcolor):
        node_id = str(uri)
        label = get_rdfs_label(g, uri)
        comment = get_rdfs_comment(g, uri)

        existing_node = dot.get_node(node_id)
        if existing_node:
            node = existing_node[0]
        else:
            node = pydot.Node(node_id)
            dot.add_node(node)

        node.set("shape", shape)
        node.set("style", "filled")
        node.set("fillcolor", fillcolor)
        node.set("color", color)
        node.set("label", label)
        if comment:
            node.set("tooltip", comment)

        return node

    # --- 1) Identify all Classes ---
    owl_classes = set(g.subjects(RDF.type, OWL.Class))
    for cls in owl_classes:
        add_or_update_node(cls, shape="box", color="black", fillcolor="lightblue")

    # --- 2) subClassOf edges ---
    # We'll store edges in a set to ensure no duplicates
    edge_set = set()
    for subclass, superclass in g.subject_objects(RDFS.subClassOf):
        if (subclass, RDF.type, OWL.Class) in g:
            sub_id = str(subclass)
            sup_id = str(superclass)

            add_or_update_node(subclass, "box", "black", "lightblue")
            add_or_update_node(superclass, "box", "black", "lightblue")

            edge_tuple = (sub_id, sup_id, "subClassOf")
            if edge_tuple not in edge_set:
                edge_set.add(edge_tuple)
                edge = pydot.Edge(
                    sub_id, sup_id,
                    label="subClassOf",
                    style="dotted",
                    color="blue"
                )
                dot.add_edge(edge)

    # --- 3) ObjectProperties: domain->range edges ---
    object_properties = set(g.subjects(RDF.type, OWL.ObjectProperty))

    for prop in object_properties:
        prop_label = get_rdfs_label(g, prop)
        # domain(s)
        for domain in g.objects(prop, RDFS.domain):
            add_or_update_node(domain, shape="box", color="black", fillcolor="lightblue")
            # range(s)
            for rng in g.objects(prop, RDFS.range):
                add_or_update_node(rng, shape="box", color="black", fillcolor="lightblue")
                domain_id = str(domain)
                range_id = str(rng)
                edge_tuple = (domain_id, range_id, prop_label)
                if edge_tuple not in edge_set:
                    edge_set.add(edge_tuple)
                    dot.add_edge(pydot.Edge(
                        domain_id, range_id,
                        label=prop_label,
                        fontsize="8",
                        color="gray"
                    ))

    # --- 4) Identify Individuals ---
    individuals = set()
    for subj, _, obj in g.triples((None, RDF.type, None)):
        if obj not in (OWL.Class, OWL.ObjectProperty, RDFS.Class):
            individuals.add(subj)

    for ind in individuals:
        add_or_update_node(ind, "ellipse", "black", "lightyellow")

    # -- 4b) rdf:type edges for individuals to classes
    for ind in individuals:
        ind_id = str(ind)
        for obj_class in g.objects(ind, RDF.type):
            if (obj_class, RDF.type, OWL.Class) in g:
                cls_id = str(obj_class)
                edge_tuple = (ind_id, cls_id, "rdf:type")
                if edge_tuple not in edge_set:
                    edge_set.add(edge_tuple)
                    dot.add_edge(pydot.Edge(
                        ind_id, cls_id,
                        label="rdf:type",
                        color="green"
                    ))

    # --- 5) Instance-level ObjectProperties
    for ind in individuals:
        ind_id = str(ind)
        for prop, obj in g.predicate_objects(ind):
            if (prop, RDF.type, OWL.ObjectProperty) in g:
                add_or_update_node(obj, "ellipse", "black", "lightyellow")
                prop_label = get_rdfs_label(g, prop)
                obj_id = str(obj)
                edge_tuple = (ind_id, obj_id, prop_label)
                if edge_tuple not in edge_set:
                    edge_set.add(edge_tuple)
                    dot.add_edge(pydot.Edge(
                        ind_id, obj_id,
                        label=prop_label,
                        color="green"
                    ))

    dot.write_raw(output_dot)
    print(f"[INFO] Wrote robust, deduplicated .dot to: {output_dot}")
