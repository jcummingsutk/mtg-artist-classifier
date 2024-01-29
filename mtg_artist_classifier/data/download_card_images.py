import os

import yaml
from mtgsdk import Card

from mtg_artist_classifier.data.utils import download_images, get_cards_for_artist


def download_unique_cards(artist: str):
    card_list_guay = get_cards_for_artist(artist)

    cards_with_multiverse_id = [
        card for card in card_list_guay if card.multiverse_id is not None
    ]
    cards_to_download: list[Card] = []
    card_names_added = []

    for card in cards_with_multiverse_id:
        if card.name not in card_names_added:
            cards_to_download.append(card)
            card_names_added.append(card.name)
    artists_data_folder = os.path.join("data", "card_images", artist)
    os.makedirs(artists_data_folder, exist_ok=True)
    download_images(
        cards_to_download,
        artists_data_folder,
    )


def main():
    with open("dvc_params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
        artists = params["artists"]
        print(artists)
    for artist in artists:
        download_unique_cards(artist)


if __name__ == "__main__":
    main()
