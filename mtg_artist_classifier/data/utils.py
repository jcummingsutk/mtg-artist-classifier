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


def get_image_uri(input_multiverse_id: str):
    """Get the uris of image (without card border) in the input_multiverse_id from the scryfall api

    Args:
        input_multiverse_id (str): id for mtg card to get url for

    Returns:
        str: uri for the image
    """

    api_link = f"https://api.scryfall.com/cards/multiverse/{input_multiverse_id}"

    response = requests.get(api_link)
    if response.status_code == 200:
        response_json = response.json()
        image_uri = response_json["image_uris"]["art_crop"]
        time.sleep(0.1)  # API requests to wait 50-100 ms per request
        return image_uri
    return None


def download_card_images(cards: list[Card], folder_dir: str):
    """Download all images of the list of cards to the folder dir

    Args:
        cards (list[Card]): list of cards' images to download
        folder_dir (str): relative folder directory to save them in
    """

    for card in cards:
        filename = os.path.join(folder_dir, f"{card.multiverse_id}.jpg")

        image_uri = get_image_uri(card.multiverse_id)

        if image_uri is not None:
            response = requests.get(image_uri, stream=True)
            time.sleep(0.1)  # API requests to wait 50-100 ms per request

            # download the image if the status code is fine
            if response.status_code == 200:
                response.raw.decode_content = True
                with open(filename, "wb") as f:
                    shutil.copyfileobj(response.raw, f)
