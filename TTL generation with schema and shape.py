#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate RDF resources, SHACL shapes, and schema summaries from company data."""

import json
import argparse
import sys
from collections import defaultdict
from urllib.parse import quote

from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD, OWL, SKOS


# ======= CONFIG TOGGLES (edit here) =======
RUN_SHACL_VALIDATION = True          # run SHACL (requires pyshacl)
RUN_STRICT_SCHEMA    = True          # check core domain/range + inverse presence
CHECK_AGAINST_SCHEMA = None          # e.g., "expected_schema.json" or None
AUTO_NAME_OUTPUTS    = True          # derive output names from input file
DEFAULT_TTL_NAME     = "output2.ttl"  # used when AUTO_NAME_OUTPUTS=False
DEFAULT_SHAPES_NAME  = "shapes2.ttl"
DEFAULT_SCHEMA_NAME  = "schema2.json"
# ===========================================================


# --- RDF NAMESPACE SETUP ---
SDK = Namespace("http://asu.edu/semantics/SUDOKN/")
IOF_CORE = Namespace("https://spec.industrialontologies.org/ontology/core/Core/")
IOF_SCRO = Namespace("https://spec.industrialontologies.org/ontology/supplychain/SupplyChain/")
XSD_NS = Namespace("http://www.w3.org/2001/XMLSchema#")
RDFS_NS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
BFO = Namespace("http://purl.obolibrary.org/obo/")
GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
GEO_SPARQL = Namespace("http://www.opengis.net/ont/geosparql#")
SH = Namespace("http://www.w3.org/ns/shacl#")

NAICS = Namespace(str(SDK) + "naics/")  # skos:Concepts for NAICS

# ------------- helpers -------------
def uri_strip(val: str) -> str:
    """Normalize free-text text into a URI-safe slug."""
    if val is None:
        return ""
    cleaned = (
        str(val)
        .replace(" ", "-")
        .replace("/", "-")
        .replace("&", "-")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace('"', "")
        .replace("'", "")
        .replace("`", "")
    )
    return quote(cleaned, safe='-._~')

def add_label(g: Graph, s: URIRef, label: str, lang: str="en"):
    """Attach primary label variants and search tokens to an entity."""
    if not label:
        return
    g.add((s, RDFS.label, Literal(label, lang=lang)))
    g.add((s, SKOS.prefLabel, Literal(label, lang=lang)))
    g.add((s, SDK.searchLabel, Literal(str(label).lower(), lang=lang)))

def add_alt_labels(g: Graph, s: URIRef, labels):
    """Record optional SKOS alternative labels for an entity."""
    for l in (labels or []):
        g.add((s, SKOS.altLabel, Literal(l, lang="en")))

def add_inverse(g: Graph, s: URIRef, p_forward: URIRef, o: URIRef, p_inverse: URIRef=None):
    """Assert forward triple and optional inverse triple in the graph."""
    g.add((s, p_forward, o))
    if p_inverse:
        g.add((o, p_inverse, s))

def safe_int(v):
    """Convert to int when possible, otherwise return ``None``."""
    try:
        return int(v)
    except Exception:
        return None

# ------------- enrichment -------------
def enrich_company_rdf(company, g: Graph):
    """Expand a company JSON record into typed RDF resources and relationships."""
    gid_val = company.get("global_id") or company.get("GlobalID") or company.get("Global_Id")
    if not gid_val:
        return

    gid_stripped = uri_strip(gid_val)
    subj = SDK[f"{gid_stripped}-company-inst"]
    g.add((subj, RDF.type, IOF_CORE.Manufacturer))

    # Name
    name = None
    is_m = company.get("is_manufacturer")
    if isinstance(is_m, dict) and is_m.get("name"):
        name = is_m.get("name")
    else:
        name = company.get("name") or company.get("Name")
    if name:
        add_label(g, subj, name)

    # Industries
    inds = company.get('industries')
    industries = inds.get('results', []) if isinstance(inds, dict) else (inds or [])
    for ind in industries:
        if not ind:
            continue
        ind_uri = SDK[f"{uri_strip(ind)}-inst"]
        add_inverse(g, subj, SDK.suppliesToIndustry, ind_uri, SDK.suppliedByIndustry)
        class_name = f"{''.join(e for e in ind.title() if e.isalnum())}Industry"
        g.add((ind_uri, RDF.type, SDK[class_name]))
        add_label(g, ind_uri, ind)

    # Process capabilities
    pcaps = company.get('process_caps')
    process_caps = pcaps.get('results', []) if isinstance(pcaps, dict) else (pcaps or [])
    gid_short = gid_stripped.split('_')[0]
    for pc in process_caps:
        if not pc:
            continue
        pc_uri = SDK[f"{gid_short}-{uri_strip(pc)}-inst"]
        class_name = f"{''.join(e for e in pc.title() if e.isalnum())}Capability"
        g.add((pc_uri, RDF.type, SDK[class_name]))
        add_label(g, pc_uri, pc)
        add_inverse(g, subj, SDK.hasProcessCapability, pc_uri, SDK.processCapabilityOf)

    # Certifications
    certs_dict = company.get('certificates')
    certs = certs_dict.get('results', []) if isinstance(certs_dict, dict) else []
    for cert in certs:
        if not cert:
            continue
        cert_local_name = uri_strip(cert)
        cert_uri = SDK[f"{gid_stripped}-{cert_local_name}"]
        g.add((cert_uri, RDF.type, SDK[cert_local_name]))
        add_label(g, cert_uri, cert)
        add_inverse(g, subj, SDK.hasCertificate, cert_uri, SDK.certificateOf)

    # Material capabilities
    mcaps = company.get('material_caps')
    mat_caps = mcaps.get('results', []) if isinstance(mcaps, dict) else (mcaps or [])
    for mc in mat_caps:
        if not mc:
            continue
        mc_uri = SDK[f"{gid_stripped}-{uri_strip(mc)}-inst"]
        class_name = f"{''.join(e for e in mc.title() if e.isalnum())}ProcessingCapability"
        g.add((mc_uri, RDF.type, SDK[class_name]))
        add_label(g, mc_uri, mc)
        add_inverse(g, subj, SDK.hasMaterialCapability, mc_uri, SDK.materialCapabilityOf)

    # Business statuses
    bs = company.get('business_statuses')
    statuses = bs.get('results', []) if isinstance(bs, dict) else (bs or [])
    for stat in statuses:
        if not stat:
            continue
        stat_uri = SDK[f"{uri_strip(stat)}-inst"]
        class_name = f"{''.join(e for e in stat.title() if e.isalnum())}BusinessStatus"
        g.add((stat_uri, RDF.type, SDK[class_name]))
        add_label(g, stat_uri, stat)
        add_inverse(g, subj, SDK.hasOwnershipStatusClassifier, stat_uri, SDK.isOwnershipStatusOf)

    # NAICS as SKOS concepts
    naics_list = company.get('naics') or []
    for na in naics_list:
        if not na or not isinstance(na, dict):
            continue
        code = na.get('code')
        desc = na.get('desc')
        if not code:
            continue
        naics_uri = NAICS[uri_strip(code)]
        g.add((naics_uri, RDF.type, SKOS.Concept))
        add_label(g, naics_uri, f"NAICS {code}")
        if desc:
            g.add((naics_uri, SKOS.prefLabel, Literal(desc, lang="en")))
        g.add((naics_uri, SKOS.notation, Literal(str(code))))
        add_inverse(g, subj, SDK.hasPrimaryNAICSClassifier, naics_uri, SDK.primaryClassifierFor)

    # Products
    prods = company.get('products_old') or company.get('products')
    products = prods.get('results', []) if isinstance(prods, dict) else (prods or [])
    for prod in products:
        if not prod:
            continue
        prod_uri = SDK[f"{uri_strip(prod)}-product"]
        g.add((prod_uri, RDF.type, IOF_CORE.MaterialProduct))
        add_label(g, prod_uri, prod)
        add_inverse(g, subj, SDK.manufactures, prod_uri, SDK.manufacturedBy)

    # Business description
    business_desc = company.get('business_desc') or company.get('Business_Description')
    if business_desc:
        desc_uri = SDK[f"{gid_stripped}-company-description"]
        g.add((desc_uri, RDF.type, SDK.BusinessDescription))
        g.add((subj, IOF_CORE.describedBy, desc_uri))
        g.add((desc_uri, IOF_SCRO.hasTextValue, Literal(business_desc, lang="en")))

    # Number of employees
    num_employees = company.get('num_employees') or company.get('Employees')
    n_em = safe_int(num_employees)
    if n_em is not None:
        g.add((subj, SDK.hasNumberOfEmployees, Literal(n_em, datatype=XSD.integer)))

    # Web address
    url = company.get('url')
    if url:
        web_uri = SDK[f"WebAddress-{uri_strip(url)}-inst"]
        add_inverse(g, subj, SDK.hasWebAddress, web_uri, SDK.webAddressOf)
        add_label(g, web_uri, url)

    # Year founded
    year = company.get('year_founded') or company.get('Year_Founded')
    y = safe_int(year)
    if y is not None:
        g.add((subj, SDK.hasOrganizationYearOfEstablishment, Literal(y, datatype=XSD.gYear)))

    # Addresses
    addresses = company.get('addresses') or []
    for addr in addresses:
        city_val = addr.get('city')
        state_val = addr.get('state')
        zipc_val = addr.get('zip')
        line_1 = addr.get('line_1')
        lat_val = addr.get('latitude')
        long_val = addr.get('longitude')
        city_geoid = addr.get('city_geoid') or addr.get('city_dcid')
        state_geoid = addr.get('state_geoid') or addr.get('state_dcid')

        if not city_val:
            continue

        gid_short = gid_stripped.split('_')[0]
        zip_clean = str(zipc_val).strip() if zipc_val not in (None, "") else None
        if not zip_clean or zip_clean.lower() in {"unknown", "00000"}:
            continue

        zip_tag = uri_strip(zip_clean.split("-")[0])
        site_uri = SDK[f"{uri_strip(gid_short)}-site-{uri_strip(city_val)}-{zip_tag}"]
        g.add((site_uri, RDF.type, SDK.GeospatialLocation))
        add_inverse(g, subj, SDK.organizationLocatedIn, site_uri, SDK.locationOfOrganization)

        coord_uri = SDK[f"{uri_strip(gid_short)}-site-{uri_strip(city_val)}-{zip_tag}-coordinates"]
        g.add((coord_uri, RDF.type, SDK.TwoDimensionalCartesianSpatialCoordinateDatum))
        g.add((site_uri, SDK.hasSpatialCoordinates, coord_uri))
        if lat_val is not None:
            g.add((coord_uri, SDK.hasLatitudeValue, Literal(str(lat_val))))
        if long_val is not None:
            g.add((coord_uri, SDK.hasLongitudeValue, Literal(str(long_val))))

        if lat_val is not None and long_val is not None:
            try:
                lat_f = float(lat_val); lon_f = float(long_val)
                g.add((site_uri, GEO.lat, Literal(lat_f)))
                g.add((site_uri, GEO.long, Literal(lon_f)))
                wkt = f"POINT({lon_f} {lat_f})"
                g.add((site_uri, GEO_SPARQL.asWKT, Literal(wkt, datatype=GEO_SPARQL.wktLiteral)))
            except Exception:
                pass

        # state
        state_uri_ref = None
        if state_geoid:
            if str(state_geoid).startswith("geoId/"):
                state_uri_ref = URIRef(f"https://datacommons.org/place/{state_geoid}")
            elif str(state_geoid).startswith("geoId"):
                state_uri_ref = URIRef(f"https://datacommons.org/place/{state_geoid}")
            else:
                state_uri_ref = URIRef(f"https://datacommons.org/place/geoId/{state_geoid}")
            g.add((state_uri_ref, RDF.type, SDK.State))
            if state_val:
                add_label(g, state_uri_ref, state_val)
        elif state_val:
            state_uri_ref = URIRef(str(SDK[f"{uri_strip(state_val)}-State"]))
            g.add((state_uri_ref, RDF.type, SDK.State))
            add_label(g, state_uri_ref, state_val)

        # city
        if city_geoid:
            if str(city_geoid).startswith("geoId/"):
                city_node = URIRef(f"https://datacommons.org/place/{city_geoid}")
            elif str(city_geoid).startswith("geoId"):
                city_node = URIRef(f"https://datacommons.org/place/{city_geoid}")
            else:
                city_node = URIRef(f"https://datacommons.org/place/geoId/{city_geoid}")
            g.add((city_node, RDF.type, SDK.City))
            add_label(g, city_node, city_val)
            add_inverse(g, site_uri, SDK.locatedInCity, city_node, SDK.cityHasLocation)
            if state_uri_ref:
                add_inverse(g, city_node, SDK.locatedInState, state_uri_ref, SDK.stateHasCity)
        else:
            city_node = SDK[f"{uri_strip(city_val)}-City"]
            g.add((city_node, RDF.type, SDK.City))
            add_label(g, city_node, city_val)
            add_inverse(g, site_uri, SDK.locatedInCity, city_node, SDK.cityHasLocation)
            if state_uri_ref:
                add_inverse(g, city_node, SDK.locatedInState, state_uri_ref, SDK.stateHasCity)

        # zip
        zip_uri = SDK[f"{uri_strip(gid_short)}-site-{uri_strip(city_val)}-{zip_tag}-site-zip"]
        g.add((zip_uri, RDF.type, SDK.USPostalCode))
        g.add((zip_uri, SDK.hasIntegerValue, Literal(zip_clean, datatype=XSD.string)))
        add_inverse(g, site_uri, SDK.hasZIPcode, zip_uri, SDK.zipOfSite)

        # postal address text
        postal_uri = SDK[f"{uri_strip(gid_short)}-site-{uri_strip(city_val)}-{zip_tag}-postal-address"]
        g.add((postal_uri, RDF.type, SDK.PostalAddress))
        add_inverse(g, site_uri, SDK.hasPostalAddress, postal_uri, SDK.postalAddressOf)
        if line_1:
            g.add((postal_uri, IOF_SCRO.hasTextValue, Literal(line_1, lang="en")))

# ------------- tiny TBox hints -------------
def emit_basic_tbox(g: Graph):
    """Populate foundational domain/range axioms for common properties."""
    # Domains / Ranges for key props
    g.add((SDK.manufactures, RDFS.domain, IOF_CORE.Manufacturer))
    g.add((SDK.manufactures, RDFS.range, IOF_CORE.MaterialProduct))
    g.add((SDK.manufacturedBy, RDFS.domain, IOF_CORE.MaterialProduct))
    g.add((SDK.manufacturedBy, RDFS.range, IOF_CORE.Manufacturer))

    g.add((SDK.hasProcessCapability, RDFS.domain, IOF_CORE.Manufacturer))
    g.add((SDK.hasProcessCapability, RDFS.range, SDK.ProcessCapability))
    g.add((SDK.processCapabilityOf, RDFS.domain, SDK.ProcessCapability))
    g.add((SDK.processCapabilityOf, RDFS.range, IOF_CORE.Manufacturer))

    g.add((SDK.organizationLocatedIn, RDFS.domain, IOF_CORE.Manufacturer))
    g.add((SDK.organizationLocatedIn, RDFS.range, SDK.GeospatialLocation))
    g.add((SDK.locationOfOrganization, RDFS.domain, SDK.GeospatialLocation))
    g.add((SDK.locationOfOrganization, RDFS.range, IOF_CORE.Manufacturer))

    g.add((SDK.suppliesToIndustry, RDFS.domain, IOF_CORE.Manufacturer))
    g.add((SDK.suppliesToIndustry, RDFS.range, SDK.Industry))
    g.add((SDK.suppliedByIndustry, RDFS.domain, SDK.Industry))
    g.add((SDK.suppliedByIndustry, RDFS.range, IOF_CORE.Manufacturer))

    g.add((SDK.hasPrimaryNAICSClassifier, RDFS.domain, IOF_CORE.Manufacturer))
    g.add((SDK.hasPrimaryNAICSClassifier, RDFS.range, SKOS.Concept))
    g.add((SDK.primaryClassifierFor, RDFS.domain, SKOS.Concept))
    g.add((SDK.primaryClassifierFor, RDFS.range, IOF_CORE.Manufacturer))

    g.add((SDK.locatedInCity, RDFS.domain, SDK.GeospatialLocation))
    g.add((SDK.locatedInCity, RDFS.range, SDK.City))
    g.add((SDK.cityHasLocation, RDFS.domain, SDK.City))
    g.add((SDK.cityHasLocation, RDFS.range, SDK.GeospatialLocation))

# ------------- SHACL shapes (compact, pragmatic) -------------
def build_shapes_graph() -> Graph:
    """Construct SHACL shapes covering manufacturers, sites, products, and NAICS."""
    s = Graph()
    s.bind("sdk", SDK); s.bind("iof-core", IOF_CORE); s.bind("iof-scro", IOF_SCRO)
    s.bind("xsd", XSD_NS); s.bind("rdfs", RDFS_NS); s.bind("bfo", BFO)
    s.bind("skos", SKOS); s.bind("geo", GEO); s.bind("geosparql", GEO_SPARQL); s.bind("sh", SH)

    man_shape = BNode()
    s.add((man_shape, RDF.type, SH.NodeShape))
    s.add((man_shape, SH.targetClass, IOF_CORE.Manufacturer))

    def prop(shape, path, node_kind=None, datatype=None, minc=None, maxc=None, name=None):
        """Helper to add a property constraint to the SHACL shapes graph."""
        bp = BNode()
        s.add((shape, SH.property, bp))
        s.add((bp, SH.path, path))
        if node_kind: s.add((bp, SH.nodeKind, node_kind))
        if datatype: s.add((bp, SH.datatype, datatype))
        if minc is not None: s.add((bp, SH.minCount, Literal(minc, datatype=XSD.integer)))
        if maxc is not None: s.add((bp, SH.maxCount, Literal(maxc, datatype=XSD.integer)))
        if name: s.add((bp, SH.name, Literal(name)))

    prop(man_shape, RDFS.label, datatype=XSD.string, minc=0, name="label")
    prop(man_shape, SDK.manufactures, node_kind=SH.IRI, minc=0, name="manufactures")
    prop(man_shape, SDK.hasProcessCapability, node_kind=SH.IRI, minc=0, name="process capability")
    prop(man_shape, SDK.organizationLocatedIn, node_kind=SH.IRI, minc=0, name="site")
    prop(man_shape, SDK.hasPrimaryNAICSClassifier, node_kind=SH.IRI, minc=0, name="naics")

    site_shape = BNode()
    s.add((site_shape, RDF.type, SH.NodeShape))
    s.add((site_shape, SH.targetClass, SDK.GeospatialLocation))
    prop(site_shape, GEO.lat, datatype=XSD.decimal, minc=0, name="lat")
    prop(site_shape, GEO.long, datatype=XSD.decimal, minc=0, name="long")
    prop(site_shape, SDK.locatedInCity, node_kind=SH.IRI, minc=0, name="city")
    prop(site_shape, SDK.hasZIPcode, node_kind=SH.IRI, minc=0, name="zip")

    prod_shape = BNode()
    s.add((prod_shape, RDF.type, SH.NodeShape))
    s.add((prod_shape, SH.targetClass, IOF_CORE.MaterialProduct))
    prop(prod_shape, RDFS.label, datatype=XSD.string, minc=0, name="label")

    naics_shape = BNode()
    s.add((naics_shape, RDF.type, SH.NodeShape))
    s.add((naics_shape, SH.targetClass, SKOS.Concept))
    prop(naics_shape, SKOS.notation, datatype=XSD.string, minc=1, name="code")
    prop(naics_shape, SKOS.prefLabel, datatype=XSD.string, minc=0, name="name")

    return s

# ------------- schema.json extraction for LLM -------------
def export_schema_summary(g: Graph, out_path="schema.json"):
    """Aggregate property usage statistics into a JSON artifact for tooling.

    Parameters
    ----------
    g : Graph
        RDF graph containing both ABox and TBox assertions to summarize.
    out_path : str, default "schema.json"
        Destination filepath for the generated schema summary.
    """
    props = defaultdict(lambda: {"domain": None, "range": None, "labels": set(), "examples": [], "count": 0})
    for s, p, o in g:
        props[str(p)]["count"] += 1
    for p, _, o in g.triples((None, RDFS.domain, None)):
        props[str(p)]["domain"] = str(o)
    for p, _, o in g.triples((None, RDFS.range, None)):
        props[str(p)]["range"] = str(o)
    for p, _, l in g.triples((None, RDFS.label, None)):
        props[str(p)]["labels"].add(str(l))

    for s, p, o in g:
        ps = str(p)
        ex = props[ps]["examples"]
        if len(ex) < 3:
            ex.append([str(s), ps, str(o)])

    inv_map = {
        str(SDK.manufactures): str(SDK.manufacturedBy),
        str(SDK.hasProcessCapability): str(SDK.processCapabilityOf),
        str(SDK.organizationLocatedIn): str(SDK.locationOfOrganization),
        str(SDK.suppliesToIndustry): str(SDK.suppliedByIndustry),
        str(SDK.locatedInCity): str(SDK.cityHasLocation),
        str(SDK.hasPrimaryNAICSClassifier): str(SDK.primaryClassifierFor),
    }

    data = {
        "namespaces": {
            "sdk": str(SDK), "iof-core": str(IOF_CORE), "iof-scro": str(IOF_SCRO),
            "skos": str(SKOS), "rdfs": str(RDFS), "xsd": str(XSD),
            "geo": str(GEO), "geosparql": str(GEO_SPARQL)
        },
        "properties": {
            k: {
                "domain": v["domain"],
                "range": v["range"],
                "labels": sorted(v["labels"]),
                "examples": v["examples"],
                "count": v["count"],
                "inverse": inv_map.get(k)
            }
            for k, v in sorted(props.items(), key=lambda kv: -kv[1]["count"])
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ------------- validation utilities -------------
def validate_with_shacl(ttl_path: str, shapes_path: str) -> bool:
    """Run pySHACL validation and emit human-readable status logging.

    Parameters
    ----------
    ttl_path : str
        Filesystem path to the generated Turtle data graph.
    shapes_path : str
        Filesystem path to the accompanying SHACL shapes document.

    Returns
    -------
    bool
        ``True`` if validation passes or is skipped, else ``False``.
    """
    try:
        from pyshacl import validate
    except ImportError:
        print("[warn] pyshacl not installed; skip SHACL validation. Install with: pip install pyshacl")
        return True

    data_g = Graph(); data_g.parse(ttl_path, format="turtle")
    shapes_g = Graph(); shapes_g.parse(shapes_path, format="turtle")

    conforms, report_graph, report_text = validate(
        data_graph=data_g,
        shacl_graph=shapes_g,
        inference='rdfs',
        abort_on_first=False,
        meta_shacl=False,
        advanced=True,
        js=False
    )
    print("[shacl] conforms:", conforms)
    if not conforms:
        print(report_text)
    return bool(conforms)

def strict_schema_checks(g: Graph) -> list:
    """Verify that foundational predicates expose the expected TBox structure.

    Parameters
    ----------
    g : Graph
        RDF graph containing the emitted TBox assertions.

    Returns
    -------
    list[str]
        Collection of human-readable errors describing missing axioms.
    """
    reqs = [
        (SDK.manufactures, IOF_CORE.Manufacturer, IOF_CORE.MaterialProduct, SDK.manufacturedBy),
        (SDK.hasProcessCapability, IOF_CORE.Manufacturer, SDK.ProcessCapability, SDK.processCapabilityOf),
        (SDK.organizationLocatedIn, IOF_CORE.Manufacturer, SDK.GeospatialLocation, SDK.locationOfOrganization),
        (SDK.suppliesToIndustry, IOF_CORE.Manufacturer, SDK.Industry, SDK.suppliedByIndustry),
        (SDK.locatedInCity, SDK.GeospatialLocation, SDK.City, SDK.cityHasLocation),
        (SDK.hasPrimaryNAICSClassifier, IOF_CORE.Manufacturer, SKOS.Concept, SDK.primaryClassifierFor),
    ]
    errors = []
    for p, dom, rng, inv in reqs:
        if (p, RDFS.domain, dom) not in g:
            errors.append(f"Missing domain: {p.n3()} rdfs:domain {dom.n3()}")
        if (p, RDFS.range, rng) not in g:
            errors.append(f"Missing range: {p.n3()} rdfs:range {rng.n3()}")
        if inv and (inv, OWL.inverseOf, p) not in g and (p, OWL.inverseOf, inv) not in g:
            # allow absence of owl:inverseOf if both properties exist; still warn
            if inv not in list(g.subjects()) and inv not in list(g.predicates()):
                errors.append(f"Missing inverse property declaration (owl:inverseOf or presence of {inv.n3()}) for {p.n3()}")
    return errors

def compare_with_schema_json(g: Graph, schema_json_path: str) -> list:
    """Cross-check emitted predicates with a prior schema fingerprint.

    Parameters
    ----------
    g : Graph
        Graph capturing the generated TBox axioms.
    schema_json_path : str
        Filepath to the JSON schema summary previously produced.

    Returns
    -------
    list[str]
        Descriptions of mismatched or absent domains/ranges.
    """
    try:
        with open(schema_json_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except Exception as e:
        return [f"Failed to read schema.json: {e}"]

    mismatches = []
    props = schema.get("properties", {})
    for p_iri, meta in props.items():
        p_ref = URIRef(p_iri)
        dom = meta.get("domain")
        rng = meta.get("range")
        if dom and (p_ref, RDFS.domain, URIRef(dom)) not in g:
            mismatches.append(f"{p_iri} domain mismatch or missing in TTL: expected {dom}")
        if rng and (p_ref, RDFS.range, URIRef(rng)) not in g:
            mismatches.append(f"{p_iri} range mismatch or missing in TTL: expected {rng}")
    return mismatches

# ------------- main -------------
def main():
    """CLI entrypoint orchestrating TTL publication, SHACL generation, and validation."""
    parser = argparse.ArgumentParser(description="JSON → RDF Turtle + SHACL + schema.json (toggles in code)")
    parser.add_argument("-i", "--input", required=True, help="Path to JSON array of companies")
    args = parser.parse_args()

    # auto-name outputs from input file if enabled
    if AUTO_NAME_OUTPUTS:
        # base = args.input.rsplit(".", 1)[0]
        base = "output"
        ttl_path    = f"{base}.ttl"
        shapes_path = f"{base}.shapes.ttl"
        schema_path = f"{base}.schema.json"
    else:
        ttl_path    = DEFAULT_TTL_NAME
        shapes_path = DEFAULT_SHAPES_NAME
        schema_path = DEFAULT_SCHEMA_NAME

    # build graph
    g = Graph()
    g.bind("sdk", SDK);
    g.bind("iof-core", IOF_CORE);
    g.bind("iof-scro", IOF_SCRO)
    g.bind("xsd", XSD_NS);
    g.bind("rdfs", RDFS_NS);
    g.bind("bfo", BFO)
    g.bind("skos", SKOS);
    g.bind("geo", GEO);
    g.bind("geosparql", GEO_SPARQL)
    g.bind("owl", OWL)

    with open(args.input, "r", encoding="utf-8") as f:
        companies = json.load(f)
    for comp in companies:
        enrich_company_rdf(comp, g)

    emit_basic_tbox(g)

    # write TTL
    with open(ttl_path, "w", encoding="utf-8") as fout:
        fout.write('@base <http://asu.edu/semantics/SUDOKN/> .\n')
        fout.write(g.serialize(format="turtle"))

    # SHACL
    shapes_g = build_shapes_graph()
    shapes_g.serialize(destination=shapes_path, format="turtle")

    # schema summary JSON
    export_schema_summary(g, out_path=schema_path)

    # -------- validation phase (driven by toggles) --------
    ok = True
    if RUN_STRICT_SCHEMA:
        tbox_errors = strict_schema_checks(g)
        if tbox_errors:
            ok = False
            print("[strict-schema] FAIL:")
            for e in tbox_errors:
                print(" -", e)
        else:
            print("[strict-schema] OK")

    if CHECK_AGAINST_SCHEMA:
        mismatches = compare_with_schema_json(g, CHECK_AGAINST_SCHEMA)
        if mismatches:
            ok = False
            print("[schema-compare] FAIL:")
            for m in mismatches:
                print(" -", m)
        else:
            print("[schema-compare] OK")

    if RUN_SHACL_VALIDATION:
        if not validate_with_shacl(ttl_path, shapes_path):
            ok = False

    if not ok:
        import sys
        sys.exit(1)
    else:
        print("[done] Generated:", ttl_path, shapes_path, schema_path)

if __name__ == "__main__":
    main()
