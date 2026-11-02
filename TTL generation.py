#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight JSON-to-RDF exporter without SHACL or schema generation."""

import json
import argparse
from urllib.parse import quote

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

# --- RDF NAMESPACE SETUP ---
SDK = Namespace("http://asu.edu/semantics/SUDOKN/")
IOF_CORE = Namespace("https://spec.industrialontologies.org/ontology/core/Core/")
IOF_SCRO = Namespace("https://spec.industrialontologies.org/ontology/supplychain/SupplyChain/")
XSD_NS = Namespace("http://www.w3.org/2001/XMLSchema#")
RDFS_NS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
BFO = Namespace("http://purl.obolibrary.org/obo/")

def uri_strip(val: str) -> str:
    """Convert arbitrary text into a URI-safe identifier fragment."""
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

def enrich_company_rdf(company, g):
    """Populate the RDF graph with individuals derived from a company record.

    Parameters
    ----------
    company : dict
        Source JSON object describing the company metadata.
    g : Graph
        RDFLib graph that accumulates triples for export.
    """
    # Global/company subject
    gid_val = company.get("global_id") or company.get("GlobalID") or company.get("Global_Id")
    if not gid_val:
        return
    gid_stripped = uri_strip(gid_val)
    subj = SDK[f"{gid_stripped}-company-inst"]
    g.add((subj, RDF.type, IOF_CORE.Manufacturer))

    # Name (prefer is_manufacturer.name → fallback to name/Name)
    name = None
    is_m = company.get("is_manufacturer")
    if isinstance(is_m, dict) and is_m.get("name"):
        name = is_m.get("name")
    else:
        name = company.get("name") or company.get("Name")
    if name:
        g.add((subj, RDFS_NS.label, Literal(name)))

    # Industries
    inds = company.get('industries')
    industries = inds.get('results', []) if isinstance(inds, dict) else (inds or [])
    for ind in industries:
        if not ind:
            continue
        ind_uri = SDK[f"{uri_strip(ind)}-inst"]
        g.add((subj, SDK.suppliesToIndustry, ind_uri))
        class_name = f"{''.join(e for e in ind.title() if e.isalnum())}Industry"
        g.add((ind_uri, RDF.type, SDK[class_name]))
        g.add((ind_uri, RDFS_NS.label, Literal(ind)))

    # Process capabilities
    pcaps = company.get('process_caps')
    process_caps = pcaps.get('results', []) if isinstance(pcaps, dict) else (pcaps or [])
    for pc in process_caps:
        if not pc:
            continue
        gid_short = gid_stripped.split('_')[0]
        pc_uri = SDK[f"{gid_short}-{uri_strip(pc)}-inst"]
        class_name = f"{''.join(e for e in pc.title() if e.isalnum())}Capability"
        g.add((pc_uri, RDF.type, SDK[class_name]))
        g.add((pc_uri, RDFS_NS.label, Literal(pc)))
        g.add((subj, SDK.hasProcessCapability, pc_uri))

    # Certifications
    certs_dict = company.get('certificates')
    certs = certs_dict.get('results', []) if isinstance(certs_dict, dict) else []
    for cert in certs:
        if not cert:
            continue
        cert_uri = SDK[f"{gid_stripped}-{uri_strip(cert)}"]
        g.add((cert_uri, RDF.type, SDK[uri_strip(cert)]))
        g.add((cert_uri, RDFS_NS.label, Literal(cert)))
        g.add((subj, SDK.hasCertificate, cert_uri))

    # Material capabilities
    mcaps = company.get('material_caps')
    mat_caps = mcaps.get('results', []) if isinstance(mcaps, dict) else (mcaps or [])
    for mc in mat_caps:
        if not mc:
            continue
        mc_uri = SDK[f"{gid_stripped}-{uri_strip(mc)}-inst"]
        class_name = f"{''.join(e for e in mc.title() if e.isalnum())}ProcessingCapability"
        g.add((mc_uri, RDF.type, SDK[class_name]))
        g.add((mc_uri, RDFS_NS.label, Literal(mc)))
        g.add((subj, SDK.hasMaterialCapability, mc_uri))

    # Business statuses
    bs = company.get('business_statuses')
    statuses = bs.get('results', []) if isinstance(bs, dict) else (bs or [])
    for stat in statuses:
        if not stat:
            continue
        stat_uri = SDK[f"{uri_strip(stat)}-inst"]
        class_name = f"{''.join(e for e in stat.title() if e.isalnum())}BusinessStatus"
        g.add((stat_uri, RDF.type, SDK[class_name]))
        g.add((stat_uri, RDFS_NS.label, Literal(stat)))
        g.add((subj, SDK.hasOwnershipStatusClassifier, stat_uri))

    # NAICS
    naics_list = company.get('naics') or []
    for naics in naics_list:
        if not naics or not isinstance(naics, dict):
            continue
        code = naics.get('code')
        desc = naics.get('desc')
        if code:
            naics_uri = SDK[f"naics-{uri_strip(code)}-inst"]
            class_uri = SDK[f"NAICS{uri_strip(code)}"]
            g.add((naics_uri, RDF.type, class_uri))
            g.add((subj, SDK.hasPrimaryNAICSClassifier, naics_uri))
            if desc:
                g.add((naics_uri, SDK.hasNAICSTextValue, Literal(desc)))
            g.add((naics_uri, SDK.hasNAICSCodeValue, Literal(str(code), datatype=XSD.string)))

    # Products
    prods = company.get('products_old') or company.get('products')
    products = prods.get('results', []) if isinstance(prods, dict) else (prods or [])
    product_uris = []
    for prod in products:
        if not prod:
            continue
        prod_uri = SDK[f"{uri_strip(prod)}-product"]
        product_uris.append(prod_uri)
        g.add((prod_uri, RDFS_NS.label, Literal(prod)))
        g.add((prod_uri, RDF.type, IOF_CORE.MaterialProduct))
    for prod_uri in product_uris:
        g.add((subj, SDK.manufactures, prod_uri))

    # Business description
    business_desc = company.get('business_desc') or company.get('Business_Description')
    if business_desc:
        desc_uri = SDK[f"{gid_stripped}-company-description"]
        g.add((subj, IOF_CORE.describedBy, desc_uri))
        g.add((desc_uri, IOF_SCRO.hasTextValue, Literal(business_desc)))
        g.add((desc_uri, RDF.type, SDK.BusinessDescription))

    # Number of employees
    num_employees = company.get('num_employees') or company.get('Employees')
    if num_employees is not None:
        g.add((subj, SDK.hasNumberOfEmployees, Literal(str(num_employees), datatype=XSD.string)))

    # Web address
    url = company.get('url')
    if url:
        web_uri = SDK[f"WebAddress-{uri_strip(url)}-inst"]
        g.add((subj, SDK.hasWebAddress, web_uri))

    # Year founded
    year = company.get('year_founded') or company.get('Year_Founded')
    if year is not None:
        try:
            g.add((subj, SDK.hasOrganizationYearOfEstablishment, Literal(int(year), datatype=XSD.integer)))
        except Exception:
            pass

    # Addresses (NO lookups; use what JSON provides)
    addresses = company.get('addresses') or []
    for addr in addresses:
        city_val = addr.get('city')
        state_val = addr.get('state')
        zipc_val = addr.get('zip')
        line_1 = addr.get('line_1')
        lat_val = addr.get('latitude')
        long_val = addr.get('longitude')
        city_geoid = addr.get('city_geoid') or addr.get('city_dcid')  # flexible
        state_geoid = addr.get('state_geoid') or addr.get('state_dcid')  # flexible

        if not city_val:
            continue

        gid_short = gid_stripped.split('_')[0]
        zip_clean = str(zipc_val).strip() if zipc_val not in (None, "") else None

        # Skip if no usable ZIP (we don't try to discover it)
        if not zip_clean or zip_clean.lower() in {"unknown", "00000"}:
            continue

        # site
        zip_tag = uri_strip(zip_clean.split("-")[0])
        site_uri = SDK[f"{uri_strip(gid_short)}-site-{uri_strip(city_val)}-{zip_tag}"]
        g.add((subj, SDK.organizationLocatedIn, site_uri))
        g.add((site_uri, RDF.type, SDK.GeospatialLocation))

        # coords
        coord_uri = SDK[f"{uri_strip(gid_short)}-site-{uri_strip(city_val)}-{zip_tag}-coordinates"]
        g.add((site_uri, SDK.hasSpatialCoordinates, coord_uri))
        g.add((coord_uri, RDF.type, SDK.TwoDimensionalCartesianSpatialCoordinateDatum))
        if lat_val is not None:
            g.add((coord_uri, SDK.hasLatitudeValue, Literal(str(lat_val))))
        if long_val is not None:
            g.add((coord_uri, SDK.hasLongitudeValue, Literal(str(long_val))))

        # state
        state_uri = None
        if state_geoid:
            # If JSON already carries Data Commons style ids, build dc URL
            if str(state_geoid).startswith("geoId/"):
                state_uri = f"https://datacommons.org/place/{state_geoid}"
            elif str(state_geoid).startswith("geoId"):
                state_uri = f"https://datacommons.org/place/{state_geoid}"
            else:
                # accept raw code and still form a dc URL if it looks like a geoId
                state_uri = f"https://datacommons.org/place/geoId/{state_geoid}"
            g.add((URIRef(state_uri), RDF.type, SDK.State))
            if state_val:
                g.add((URIRef(state_uri), RDFS_NS.label, Literal(state_val)))
        elif state_val:
            state_uri = str(SDK[f"{uri_strip(state_val)}-State"])
            g.add((URIRef(state_uri), RDF.type, SDK.State))
            g.add((URIRef(state_uri), RDFS_NS.label, Literal(state_val)))

        # city
        city_uri = None
        if city_geoid:
            if str(city_geoid).startswith("geoId/"):
                city_uri = f"https://datacommons.org/place/{city_geoid}"
            elif str(city_geoid).startswith("geoId"):
                city_uri = f"https://datacommons.org/place/{city_geoid}"
            else:
                city_uri = f"https://datacommons.org/place/geoId/{city_geoid}"
            g.add((URIRef(city_uri), RDF.type, SDK.City))
            g.add((URIRef(city_uri), RDFS_NS.label, Literal(city_val)))
            g.add((site_uri, SDK.locatedInCity, URIRef(city_uri)))
            if state_uri:
                g.add((URIRef(city_uri), SDK.locatedInState, URIRef(state_uri)))
        else:
            # fallback local city individual
            local_city = SDK[f"{uri_strip(city_val)}-City"]
            g.add((local_city, RDF.type, SDK.City))
            g.add((local_city, RDFS_NS.label, Literal(city_val)))
            g.add((site_uri, SDK.locatedInCity, local_city))
            if state_uri:
                g.add((local_city, SDK.locatedInState, URIRef(state_uri)))

        # zip
        zip_uri = SDK[f"{uri_strip(gid_short)}-site-{uri_strip(city_val)}-{zip_tag}-site-zip"]
        g.add((site_uri, SDK.hasZIPcode, zip_uri))
        g.add((zip_uri, RDF.type, SDK.USPostalCode))
        g.add((zip_uri, SDK.hasIntegerValue, Literal(zip_clean, datatype=XSD.string)))

        # postal address text
        postal_uri = SDK[f"{uri_strip(gid_short)}-site-{uri_strip(city_val)}-{zip_tag}-postal-address"]
        g.add((site_uri, SDK.hasPostalAddress, postal_uri))
        g.add((postal_uri, RDF.type, SDK.PostalAddress))
        if line_1:
            g.add((postal_uri, IOF_SCRO.hasTextValue, Literal(line_1)))

def main():
    """CLI front-end for generating Turtle from the enriched RDF graph.

    Parses command-line arguments, reads the input JSON array, enriches the
    RDF graph, and writes serialized Turtle to disk.
    """
    parser = argparse.ArgumentParser(description="Pure JSON → RDF Turtle (no API calls)")
    parser.add_argument("-i", "--input", required=True, help="Path to JSON array of companies")
    parser.add_argument("-o", "--output", default="output2.ttl", help="Output TTL file")
    args = parser.parse_args()

    g = Graph()
    g.bind("sdk", SDK)
    g.bind("iof-core", IOF_CORE)
    g.bind("iof-scro", IOF_SCRO)
    g.bind("xsd", XSD_NS)
    g.bind("rdfs", RDFS_NS)
    g.bind("bfo", BFO)

    with open(args.input, "r", encoding="utf-8") as f:
        companies = json.load(f)

    for comp in companies:
        enrich_company_rdf(comp, g)

    with open(args.output, "w", encoding="utf-8") as fout:
        fout.write('@base <http://asu.edu/semantics/SUDOKN/> .\n')
        fout.write(g.serialize(format="turtle"))

if __name__ == "__main__":
    main()
