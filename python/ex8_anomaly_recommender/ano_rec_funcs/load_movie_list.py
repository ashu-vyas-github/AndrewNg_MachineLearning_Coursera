from os.path import join


def load_movie_list():
    """
    Reads the fixed movie list in movie_ids.txt and returns a list of movie names.

    Returns
    -------
    movie_names : list
        A list of strings, representing all movie names.
    """
    # Read the fixed movieulary list
    with open('./movie_ids.txt',  encoding='ISO-8859-1') as fid:
        movies = fid.readlines()

    movie_names = []
    for movie in movies:
        parts = movie.split()
        movie_names.append(' '.join(parts[1:]).strip())

    return movie_names
