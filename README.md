# System Design

## GCS / Vertex
I have decided that GCS is a better fit due to the nature of cold storage - we likely value cost savings in the cold tier more than efficient lookup so we are using normal cloud storage rather than Vertex AI, which enables more efficient lookup but much higher cost. The code is configured to run with a service account I have created (when running simulate_temperature).

## TODO

server.py likely does not need to run on Flask anymore since we have reverted to entirely in Python. It can just be functions for storing and retrieving, with a basic scoring mechanism. We should also have a migration method that retrieves an embedding from Tier X, moves it to Tier Y. This will also require an eviction policy, or an eviction function to kick out the lowest scored document within a tier.