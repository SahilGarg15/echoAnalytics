# modules/query_patterns.py

# Synonyms for better query recognition
SYNONYMS = {
    "population": ["population", "pop.", "number of people", "inhabitants"],
    "literacy": ["literacy", "education rate", "educated people", "literacy rate"],
    "area": ["area", "size", "sq km", "square kilometers"],
    "density": ["density", "population density", "people per sq km"],
    "growth_rate": ["growth", "growth rate", "annual growth"],
}

# Normalization mapping for dataset column names
COLUMN_MAPPING = {
    "population": "population",
    "pop.": "population",
    "number of people": "population",
    "inhabitants": "population",

    "literacy": "literacy_rate",
    "education rate": "literacy_rate",
    "educated people": "literacy_rate",
    "literacy rate": "literacy_rate",

    "area": "area_km2",
    "size": "area_km2",
    "sq km": "area_km2",
    "square kilometers": "area_km2",

    "density": "pop_density",
    "population density": "pop_density",
    "people per sq km": "pop_density",

    "growth": "growth_rate",
    "growth rate": "growth_rate",
    "annual growth": "growth_rate",
}

# Regex patterns for query detection
QUERY_PATTERNS = {
    "last_n_rows": r"(last|show)\s+(\d+)\s+rows",
    "population_query": r"(population|pop\.|number of people|inhabitants)\s+of\s+([\w\s]+)",
    "literacy_query": r"(literacy|education rate|educated people|literacy rate)\s+of\s+([\w\s]+)",
    "area_query": r"(area|size|sq km|square kilometers)\s+of\s+([\w\s]+)",
    "density_query": r"(density|population density|people per sq km)\s+of\s+([\w\s]+)",
    "growth_query": r"(growth|growth rate|annual growth)\s+of\s+([\w\s]+)",
}
