from string import Template


main_prompt = Template("""Create a SPARQL query that can retrieve the values that are the answer to this question: "${question}".
Your query shall be returned as is with no other text than that of the query itself. The query is based on this template:
```
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
PREFIX geo: <http://statistics.gov.scot/id/statistical-geography>

SELECT ?values
WHERE {
  ?obs qb:dataSet DATASET_URI.
  ?obs MEASURE_URI ?values.
  ?obs DIMENSION_URI DIMENSION_VALUE_URI.
}
```
Replace the DATASET_URI, MEASURE_URI, DIMENSION_URI, and DIMENSION_VALUE_URI with the available URIs presented below as
you see fit. Feel free to add more lines if needed, that follow the same structure.
Information about the URIs that can be used in the query:

""")


dims_and_measures_narrowing_prompt = Template("""
Given the question: "${question}", which of the following dimensions are needed to answer it: ${d_labels}?
Which of the following measures are needed to answer it: ${m_labels}?
Reply with the names of the dimensions and measures necessary to answer the question as they are and no symbols or
anything else, with the dimensions in a different line to the measures:

example response:
dimension1, dimension2
measure1
""")


codelist_narrowing_prompt = Template("""
Given the question: "${question}", which of the below dimension values for dimension ${dim_l} can be used to answer it?
You must select exactly one value.
dimension values:
${dim_vs}

Reply with the name of the dimension value as is and no other symbols or anything else:
""")


final_response_prompt = Template("""
Given that the user asked "${question}", and that the result returned through the API is:
${sparql_final_result}, format your response based on the result in a clear and user-friendly manner.
""")
