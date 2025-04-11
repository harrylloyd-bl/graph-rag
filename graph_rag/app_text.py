def title_text():
    text = """
        This app demonstrates some of the features involved in offering semantic search. Semantic search is search
        that uses language models to embed the content of a collection into high-dimensional space, then embeds search
        queries into the same space to find results based not just on key word results, but the _meanings_ of the
        search term and collection content. This is a hot topic in search, we've had many enquiries from
        private companies offering this kind of capability, collaborations with academics to explore it and talks
        from other organisations exploring the same thing. Whatsmore a version of it is offered in
        [Primo VE](https://knowledge.exlibrisgroup.com/Primo/Product_Documentation/020Primo_VE/Primo_VE_(English)/015_Getting_Started_with_Primo_Research_Assistant),
        the new front end for ALMA. This demo is my attempt to explain the concepts and important points to consider,
        so that when we talk about semantic search we can talk with authority, and specify clearly what we want from
        any implementation.  
        
        To create the demo I've used an old version of the British National Bibliography (BNB) linked dataset, with
        an outdated ontology. The current version of the BNB is available through a [Share Family VDE](https://bl.natbib-lod.org/)
        using an extension of the BIBFRAME ontology. The benefit of the old version was that it was immediately accessible.
        I've embedded parts of the BNB using a trio of easily accessible Transformer models through the Sentence-Transformers
        package, and use the same models to embed queries. I use UMAP as the dimensionality reduction algorithm to
        visualise results. To avoid the overheads of running an LLM locally while we lack a development environment
        I use the Hugging Face inference API, which allows access to remote open LLMs for free within usage limits.
    """
    return text
