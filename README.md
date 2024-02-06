# pepembed

Command line interface and python package for computing text-embeddings of sample metadata stored in [pephub](https://github.com/pepkit/pephub) for search-and-retrieval tasks. The purpose of this package is to handle the long-running job of downloading projects inside pephub, mining any relevant metadata from them, and then computing a rich text embedding on that data and upserting it into a vector database. We use [qdrant](https://qdrant.tech/) as our vector database for its performance and simplicity and payload capabilities.

Full documentation can be found on the [PEP documentation site](https://pep.databio.org)
