from string import Template

prefixes = """
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX qb: <http://purl.org/linked-data/cube#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX sdmx: <http://purl.org/linked-data/sdmx/2009/concept#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX void: <http://rdfs.org/ns/void#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

get_all_datasets_and_info = f"""
{prefixes}

SELECT distinct *
WHERE {{
  ?ds a qb:DataSet.
  ?ds rdfs:comment ?desc.
  ?ds rdfs:label ?label
}}
"""


get_all_properties = Template(f"""
{prefixes}

SELECT distinct *
WHERE {{ 
 <${{dataset}}> qb:structure/qb:component ?comp.
  ?comp qb:dimension ?dim.
  optional{{?dim rdfs:label ?label.}}
  optional{{?dim rdfs:comment ?comment.}}
  optional{{?comp qb:codeList ?cl.}}.
}}
""")


get_measures = Template(f"""
{prefixes}

SELECT distinct *
WHERE {{
   <${{measure_cl}}> skos:member ?measure.
   ?measure rdfs:label ?label
}}
""")


get_cls = Template(f"""
{prefixes}

SELECT distinct *
WHERE {{
 <${{cl}}> skos:member ?clvals.
 ?clvals rdfs:label ?name.
}}
""")
