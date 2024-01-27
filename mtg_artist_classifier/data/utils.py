import os
import shutil
import time

import requests
from mtgsdk import Card


def get_cards_for_artist(artist: str) -> list[Card]:
    """Uses the mtg sdk to return all cards for which the input
    artist is the artist

    Args:
        artist (str): Artist for the cards you want

    Returns:
        list[Card]: list of cards with the given artist
    """
    cards = Card.where(artist=artist).all()
    return cards


def get_image_uri(input_multiverse_id):
    """Get the uris of image (without card border) in the input_multiverse_id
    from the scryfall api
    Args:
        input_multiverse_id -- id for mtg card to get url for
    Returns:
        image_url -- list of image url
    """

    api_link = "https://api.scryfall.com/cards/multiverse/{}".format(
        input_multiverse_id
    )
    response = requests.get(api_link)
    if response.status_code == 200:
        response_json = response.json()
        try:
            image_uri = response_json["image_uris"]["art_crop"]
        except:
            return None
        time.sleep(0.1)  # API requests to wait 50-100 ms per request
        return image_uri
    else:
        return None


def download_images(cards, folder_dir):
    """Download all images of the list of cards to the folder dir
    Args:
        image_urls -- list of cards' images to download
        folder_dir -- relative folder directory to save them in
    """

    for card in cards:
        filename = os.path.join(folder_dir, f"{card.multiverse_id}.jpg")
        # Only download if the file doesn't exist or exists and is of zero size for some reason
        if (os.path.exists(filename) is False) or (
            os.path.exists(filename) is True and os.path.getsize(filename) < 10
        ):
            image_uri = get_image_uri(card.multiverse_id)

            if image_uri is not None:
                response = requests.get(image_uri, stream=True)
                time.sleep(0.1)  # API requests to wait 50-100 ms per request

                # download the image if the status code is fine
                if response.status_code == 200:
                    response.raw.decode_content = True
                    try:
                        with open(filename, "wb") as f:
                            shutil.copyfileobj(response.raw, f)
                    except:
                        pass
