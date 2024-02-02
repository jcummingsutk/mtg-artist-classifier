import os
import shutil
import time

import requests
import yaml
from mtgsdk import Card
from requests import Response


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


def copy_response_to_file(response: Response, filename: str):
    """Copies the response of an api call, which should be an image, to the filename

    Args:
        response (Response): Response with image
        filename (str): filename to save
    """
    if response.status_code == 200:
        response.raw.decode_content = True
        with open(filename, "wb") as f:
            shutil.copyfileobj(response.raw, f)


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
            copy_response_to_file(response=response, filename=filename)


def get_image_uri(input_multiverse_id: str):
    """Get the uris of image (without card border) in the input_multiverse_id from the
    scryfall api

    Args:
        input_multiverse_id (str): id for mtg card to get url for

    Returns:
        str: uri for the image
    """

    api_link = f"https://api.scryfall.com/cards/multiverse/{input_multiverse_id}"

    response = requests.get(api_link)
    if response.status_code == 200:
        response_json = response.json()
        try:
            image_uri = response_json["image_uris"]["art_crop"]
        except KeyError:
            image_uri = None
        time.sleep(0.1)  # API requests to wait 50-100 ms per request
        return image_uri
    return None


def download_card_images_for_artist(artist: str):
    """Downloads all the card images for a given artist. However, some cards
    are reprinted with the same name and art, so we will make sure not to download
    images for the same name twice

    Args:
        artist (str): artist to download images for
    """
    artist_card_list = get_cards_for_artist(artist)

    cards_with_multiverse_id = [
        card for card in artist_card_list if card.multiverse_id is not None
    ]
    cards_to_download: list[Card] = []
    card_names_added = (
        []
    )  # ensure not to add cards with the same card name twice two cards_to_download

    for card in cards_with_multiverse_id:
        if card.name not in card_names_added:
            cards_to_download.append(card)
            card_names_added.append(card.name)

    artists_data_folder = os.path.join("data", "card_images", artist)
    os.makedirs(artists_data_folder, exist_ok=True)
    download_card_images(
        cards_to_download,
        artists_data_folder,
    )


def main():
    """Gets the artist list in the dvc parameters file ane downloads
    all the card c
    """
    with open("dvc_params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
        artists = params["artists"]
        print(artists)
    for artist in artists:
        download_card_images_for_artist(artist)


if __name__ == "__main__":
    main()
