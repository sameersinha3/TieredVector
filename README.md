# System Design

## Flask Server
Documents are stored and retrieved by the flask server in server.py (this can actually just become functions we call rather than an HTTP server since the project is now entirely Python)

/retrieve endpoint works by checking hot tier documents with a given threshold, then only cascading down to mid- and cold- tier if we don't find enough documents.

## GCS / Vertex
I am currently working on Vertex vs. a standard GCS bucket. Vertex is likely more expensive but higher performance. Given that it's our cold tier, it might make more sense to use a GCS bucket to minimize cost but I will do more research into this.

