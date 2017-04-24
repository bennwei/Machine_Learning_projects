from __future__ import print_function
warned_error = False

def create_cloud(out_name, words, maxsize=120, fontname='Lobster'):
    """
    Create a word cloud when pytagcloud is installed
    :param out_name: output filename
    :param words: list of (value,str), a gensim returns (value, word)
    :param maxsize: int, optional
        Size of maximum word. The best setting for this parameter will often
        require some manual tuning for each input.
    :param fontname: str, optional, Font to use.
    :return:
    """

    try:
        from pytagcloud import create_tag_image, make_tags
    except ImportError:
        if not warned_error:
            print("Could not import pytagcloud. Skipping cloud generation!")
        return

    # gensim returns a weight between 0 and 1 for each word, while pytagcloud
    # expects an integer word count. So, we multiply by a large number and
    # round. For a visualization this is an adequate approximation.
    # We also need to flip the order as gensim returns (value, word), whilst
    # pytagcloud expects (word, value):

    words = [(w, int(v*10000)) for v, w in words]
    tags = make_tags(words, maxsize=maxsize)
    create_tag_image(tags, out_name, size=(1800, 1200), fontname=fontname)
