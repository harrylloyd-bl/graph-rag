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
        
        To create the demo I've used an old version of the British National Bibliography (BNB) linked dataset.
        I've embedded parts of the BNB using a trio of easily accessible Transformer models through the Sentence-Transformers
        package, and use the same models to embed queries. I use UMAP as the dimensionality reduction algorithm to
        visualise results. To avoid the overheads of running an LLM locally while we lack a development environment
        I use the Hugging Face inference API, which allows access to remote open LLMs for free within usage limits.
    """
    return text


def bnb_text():
    text = """
        The BNB consists of descriptions of books and journal titles published or distributed in the United Kingdom and
        Ireland since 1950, and electronic publications since 2013. It also contains forthcoming books, registered through
        the Cataloguing-in-Publication (CIP) service. The current version of the BNB is available through a [Share Family VDE](https://bl.natbib-lod.org/)
        using an extension of the BIBFRAME ontology. The version used here is older, from 2010-2011 developed with a
        previous contractor using a non-standard data model. I use the old version here because it was accessible, any discussions around the data 
        model should remember that! The data available here is a subset of 50,000 records from the Book section of the 
        BNB (no serials, or CIP forthcoming books). It is linked data, and for the purposes of this demo it's hosted
        in a Neo4J Aura database (cloud hosted, 14-day free trial). The data model pdf is included [here](./app/static/bldatamodelbook.pdf) (opens in new tab).
        The fields I've focussed on and embedded are title and any available LoC subject headings.
    """
    return text


def embedding_text():
    intro = """
        Embedding is at the core, and is one of the core ethical issues, of this demo. I want to take a moment to recap.
        We interpret the information we get through our senses to have meaning. We see an image of our home and think
        "That's my home", or hear a friend's voice and think "That's Katie". Inside our heads some complex process occurs
        to map a certain set of stimuli to memories that tell us "That input = this thing". The equivalent process for 
        computers is what we call embedding. I'll use two analogies to explain, then show how the models in use here use the process.
    """

    analogy_1 = """
        For the first explanation we'll use colours, and the R[ed] G[reen] B[lue] colour space which is one of the most commonly used.
        You specify colours by mixing certain amounts of red, green, and blue in proportion. The image below shows
        what happens if you mix Red and Green (so that we only have to think in 2D for now).
        
        <figure>
            <img src="./app/static/rg_embedding_explanation.png"
                 alt="The Red/Green colour plane">
            <figcaption>Colour plane image by Claus Wilke, https://clauswilke.com/art/post/colors-color-spaces, accessed 14/04/25</figcaption>
        </figure>
        
        Along the bottom axis is the amount of red, and along the vertical is the amount of green. What we think of as 'red' is in the bottom right
        (all red, no green), and what we think of as 'green' is in the top left (all green, no red). To explain this to a
        computer we convert the red axis to be between [0,1] and the green axis to be between [0,1]. Now we can tell the
        computer what colour we want by specifying a pair of numbers, and the computer knows when we say [1,0] we mean
        display 'Red', and likewise [0, 1] means display 'Green'. All the other combinations are possible, [0.76, 0.62]
        is supposedly a colour called 'Buddha Gold'. When we convert a colour we see to this RG colour space we have embedded
        it so the computer can understand. Colour is, of course, not objective, and your ideas of what 'red' and 'green'
        look like may differ from the chart. This is one of the pitfalls of embedding, the computer relies on what
        we have told it is the relationship between colours and numbers, and what we think can differ and change. 
    """

    analogy_2 = """
        The second analogy moves us closer to the nebulous meanings of concepts, rather than (relatively) objective colour.
        Below is a chart from the webcomic XKCD. The author, Randall Munroe, plays on the disparity between what we think
        life is made up of as children, and what it is actually (and usually more boringly) made up of as adults.
        
        <figure>
            <img src="./app/static/features_of_adulthood_2x.png"
                 alt="A graph of things you thought would come up a lot in life as a kid vs things that come up frequently as an adult">
            <figcaption>XKCD 3034: Features of Adulthood. <a href="https://creativecommons.org/licenses/by-nc/2.5/">CC BY-NC 2.5</a>, accessed 14/04/2025 and unmodified.</figcaption>
        </figure>
        
        I want you to imagine that the x- and y-axes on this graph are labelled [0,1] as in the RG analogy above. Now we could
        teach the computer (model) that things we thought would occur frequently in life as a child but very rarely occur
        in life as an adult have values around (1, 0), like quicksand, grappling hooks, and 'shoving a stick in a crocodile's
        mouth to wedge it open'. Conversely, events that we never thought would happen as a child but happen frequently as 
        an adult like 'unexplained smells or noises' and figuring out what to have for dinner have values around (0,1).
        Those we correctly identified as occuring frequently appear around (1,1), like bills, shopping, and taxes. If
        we gave the model that had been trained with all the data on the chart the term 'cooking', it would be able 
        to tell us that in this space, using these embeddings, it would have a value around (1,1), while the term 'pizza'
        would be closer to maybe (0.6, 0.6), and briefcases around (0.7, 0.3).
        
        Now the model understands how to convert these terms to points on these axes, we can ask it to compare how similar
        (how far apart) different terms are. Cool toys and power tools are similar, car chases and laundry are not. This is the
        crux of semantic search, once a model has learned how to convert a particular input (text/images/audio etc) into
        numbers, it can compare the similarity of numbers for different inputs to tell us how similar those inputs are.
        
        This graph also demonstrates, even more than the RG example, the profound effect the training data has on language
        models. This webcomic is published in English, to a mainly western, 2/3 majority male audience. Readers in
        these countries grew up with (broadly) similar backgrounds, consuming similar media, with similar experiences of
        adulthood. The humour in the comic relies on us identifying with the concepts in it, just as a language model relies
        on related training data to make accurate embeddings from a term. Viewers in different demographics, different countries,
        different upbringings, different media environments, may or may not also identify with the concepts here, or might
        choose different concepts if they were to recreate the graph. They may place concepts in different places on the graphs,
        which would generate a different set of embeddings, and would mean different concepts would be similar. The graph here
        is also relatively evenly populated, with no large gaps. If the data was all concentrated on the right hand side of the
        graph, and we asked the  model to predict the embeddings for a series of terms on the left hand side of the graph,
        its responses would be less reliable.
        
        All these things need to be considered when looking at how well an embedding model performs, and hopefully set us up 
        to ask reasonable questions of the models we're about to encounter.
    """

    models = """
        Embedding is a relatively inexpensive computational process to run (though still expensive to train). The
        Sentence Transformers package, originally developed by UKPLab at Technische Universität Darmstadt, and maintained
        by Hugging Face, provides access to a range of embedding models. I have selected three for this demo.
        Each of these models started with a different base model that was trained by giving it sentences with some of the words masked,
        then making it guess what the masked words are. The base model was then fine-tuned
        using a set of 1B pairs of similar sentences. The model was given one sentence from the pair, then had to pick the 
        correct similar pair sentence out of a random sample of the rest of the sentences in the dataset. The sources for
        the 1B pairs are provided in the [model cards](https://huggingface.co/sentence-transformers/all-mpnet-base-v2#training-data)
        on Hugging Face, but are mainly made up of similar Reddit comments, with other contributions from things like academic
        paper citations, Stack Exchange and WikiAnswers.
    """
    return intro, analogy_1, analogy_2, models
