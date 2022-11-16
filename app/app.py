import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
from pydub import AudioSegment


def main():
    page = st.sidebar.selectbox("App Selections", ["About", "Identify Audio"])
    if page == "Identify Audio":
        st.title("Bird Vocalization Classifier")
        identify()
    elif page == "About":
        st.title("Bird Vocalization Classifier")
        about()


def about():
    st.subheader("About the Model")
    st.caption("Identifying birds by sound is hard. This model was created so you don't have to!")
    st.caption("Research scientists, conservationists, and bird enthusiasts alike commonly seek to identify bird "
               "species by sound. This app uses thousands of recordings to learn the vocalizations of each bird "
               "species and gives an automated reply when you upload an audio file. ")
    st.caption("")

    st.caption("The model can classify audio recordings from 227 bird species listed below:")
    bird_names = ['Alder Flycatcher', 'American Bushtit', 'American Cliff Swallow', 'American Coot', 'American Crow',
                  'American Dusky Flycatcher', 'American Goldfinch', 'American Grey Flycatcher', 'American Kestrel',
                  'American Redstart', 'American Robin', 'American Tree Sparrow', 'American Woodcock',
                  "Anna's Hummingbird",
                  'Ash-throated Flycatcher', "Baird's Sandpiper", 'Bald Eagle', 'Baltimore Oriole', 'Barn Swallow',
                  'Barred Owl', "Bell's Sparrow", 'Belted Kingfisher', "Bewick's Wren", 'Black Phoebe',
                  'Black-and-white Warbler', 'Black-billed Magpie', 'Black-capped Chickadee',
                  'Black-chinned Hummingbird', 'Black-headed Grosbeak', 'Black-throated Blue Warbler',
                  'Black-throated Green Warbler', 'Black-throated Grey Warbler', 'Black-throated Sparrow',
                  'Blackburnian Warbler', 'Blackpoll Warbler', 'Blue Grosbeak', 'Blue Jay', 'Blue-grey Gnatcatcher',
                  'Blue-headed Vireo', 'Blue-winged Warbler', 'Bobolink', "Bonaparte's Gull", "Brewer's Blackbird",
                  "Brewer's Sparrow", 'Broad-tailed Hummingbird', 'Broad-winged Hawk', 'Brown Creeper',
                  'Brown Thrasher',
                  'Brown-headed Cowbird', 'Buff-bellied Pipit', "Bullock's Oriole", 'Cactus Wren', 'California Quail',
                  'California Scrub Jay', 'Canada Goose', 'Canada Warbler', 'Canyon Wren', 'Cape May Warbler',
                  'Carolina Wren', 'Caspian Tern', "Cassin's Finch", "Cassin's Vireo", 'Cedar Waxwing',
                  'Chestnut-sided Warbler', 'Chipping Sparrow', "Clark's Nutcracker", 'Common Goldeneye',
                  'Common Grackle',
                  'Common Loon', 'Common Merganser', 'Common Nighthawk', 'Common Redpoll', 'Common Starling',
                  'Common Tern',
                  'Common Yellowthroat', "Cooper's Hawk", 'Dark-eyed Junco', 'Downy Woodpecker', 'Eastern Bluebird',
                  'Eastern Kingbird', 'Eastern Meadowlark', 'Eastern Phoebe', 'Eastern Towhee', 'Eastern Wood Pewee',
                  'Eurasian Collared Dove', 'Eurasian Teal', 'European Herring Gull', 'Evening Grosbeak',
                  'Field Sparrow',
                  'Fish Crow', 'Gadwall', 'Golden Eagle', 'Golden-crowned Kinglet', 'Golden-crowned Sparrow',
                  'Great Blue Heron', 'Great Crested Flycatcher', 'Great Egret', 'Great Horned Owl',
                  'Great-tailed Grackle',
                  'Greater Roadrunner', 'Greater Yellowlegs', 'Green Heron', 'Green-tailed Towhee', 'Grey Catbird',
                  'Hairy Woodpecker', "Hammond's Flycatcher", 'Hermit Thrush', 'Hooded Warbler', 'Horned Grebe',
                  'Horned Lark', 'House Finch', 'House Sparrow', 'House Wren', 'Indigo Bunting', 'Juniper Titmouse',
                  'Killdeer', 'Ladder-backed Woodpecker', 'Lark Sparrow', 'Lazuli Bunting', 'Least Bittern',
                  'Least Flycatcher', 'Least Sandpiper', 'Lesser Goldfinch', 'Lesser Nighthawk', 'Lesser Yellowlegs',
                  "Lincoln's Sparrow", 'Loggerhead Shrike', 'Long-billed Curlew', 'Long-billed Dowitcher',
                  'Long-tailed Duck', 'Louisiana Waterthrush', "MacGillivray's Warbler", 'Magnolia Warbler', 'Mallard',
                  'Mangrove Warbler', 'Marsh Wren', 'Merlin', 'Mountain Chickadee', 'Mourning Dove', 'Myrtle Warbler',
                  'Northern Cardinal', 'Northern Flicker', 'Northern Mockingbird', 'Northern Parula',
                  'Northern Pintail',
                  'Northern Raven', 'Northern Shoveler', 'Northern Waterthrush', 'Olive-sided Flycatcher',
                  'Orange-crowned Warbler', 'Ovenbird', 'Pacific-slope Flycatcher', 'Pectoral Sandpiper',
                  'Peregrine Falcon', 'Phainopepla', 'Pied-billed Grebe', 'Pileated Woodpecker', 'Pine Grosbeak',
                  'Pine Siskin', 'Pine Warbler', 'Pinyon Jay', 'Plumbeous Vireo', 'Prairie Warbler', 'Purple Finch',
                  'Pygmy Nuthatch', 'Red Crossbill', 'Red Fox Sparrow', 'Red-bellied Woodpecker',
                  'Red-breasted Nuthatch',
                  'Red-eyed Vireo', 'Red-necked Phalarope', 'Red-shouldered Hawk', 'Red-tailed Hawk',
                  'Red-winged Blackbird', 'Ring-billed Gull', 'Rock Dove', 'Rock Wren', 'Rose-breasted Grosbeak',
                  'Ruby-crowned Kinglet', 'Rufous Hummingbird', 'Rusty Blackbird', 'Sand Martin', 'Savannah Sparrow',
                  "Say's Phoebe", 'Scarlet Tanager', "Scott's Oriole", 'Semipalmated Plover', 'Semipalmated Sandpiper',
                  'Short-eared Owl', 'Snow Bunting', 'Snow Goose', 'Solitary Sandpiper', 'Song Sparrow', 'Sora',
                  'Spotted Sandpiper', 'Spotted Towhee', "Steller's Jay", "Swainson's Thrush", 'Swamp Sparrow',
                  'Tree Swallow', 'Tufted Titmouse', 'Tundra Swan', 'Veery', 'Vesper Sparrow', 'Violet-green Swallow',
                  'Warbling Vireo', 'Western Grebe', 'Western Kingbird', 'Western Meadowlark', 'Western Osprey',
                  'Western Tanager', 'Western Wood Pewee', 'White-breasted Nuthatch', 'White-crowned Sparrow',
                  'White-throated Sparrow', 'Wild Turkey', 'Willow Flycatcher', "Wilson's Snipe", "Wilson's Warbler",
                  'Winter Wren', 'Wood Duck', 'Wood Thrush', "Woodhouse's Scrub Jay", 'Yellow-bellied Flycatcher',
                  'Yellow-bellied Sapsucker', 'Yellow-headed Blackbird', 'Yellow-throated Vireo']
    df = pd.DataFrame(bird_names, columns=['Species'])
    st.table(df)


def load_model():
    class_model = tf.keras.models.load_model(
        'C:/Users/allis/Bird Recognition Project/saved_models/audio_classification.hdf5')
    return class_model


model = load_model()

class_names = ['Alder Flycatcher', 'American Bushtit', 'American Cliff Swallow', 'American Coot', 'American Crow',
               'American Dusky Flycatcher', 'American Goldfinch', 'American Grey Flycatcher', 'American Kestrel',
               'American Redstart', 'American Robin', 'American Tree Sparrow', 'American Woodcock',
               "Anna's Hummingbird",
               'Ash-throated Flycatcher', "Baird's Sandpiper", 'Bald Eagle', 'Baltimore Oriole', 'Barn Swallow',
               'Barred Owl', "Bell's Sparrow", 'Belted Kingfisher', "Bewick's Wren", 'Black Phoebe',
               'Black-and-white Warbler', 'Black-billed Magpie', 'Black-capped Chickadee',
               'Black-chinned Hummingbird', 'Black-headed Grosbeak', 'Black-throated Blue Warbler',
               'Black-throated Green Warbler', 'Black-throated Grey Warbler', 'Black-throated Sparrow',
               'Blackburnian Warbler', 'Blackpoll Warbler', 'Blue Grosbeak', 'Blue Jay', 'Blue-grey Gnatcatcher',
               'Blue-headed Vireo', 'Blue-winged Warbler', 'Bobolink', "Bonaparte's Gull", "Brewer's Blackbird",
               "Brewer's Sparrow", 'Broad-tailed Hummingbird', 'Broad-winged Hawk', 'Brown Creeper', 'Brown Thrasher',
               'Brown-headed Cowbird', 'Buff-bellied Pipit', "Bullock's Oriole", 'Cactus Wren', 'California Quail',
               'California Scrub Jay', 'Canada Goose', 'Canada Warbler', 'Canyon Wren', 'Cape May Warbler',
               'Carolina Wren', 'Caspian Tern', "Cassin's Finch", "Cassin's Vireo", 'Cedar Waxwing',
               'Chestnut-sided Warbler', 'Chipping Sparrow', "Clark's Nutcracker", 'Common Goldeneye', 'Common Grackle',
               'Common Loon', 'Common Merganser', 'Common Nighthawk', 'Common Redpoll', 'Common Starling',
               'Common Tern',
               'Common Yellowthroat', "Cooper's Hawk", 'Dark-eyed Junco', 'Downy Woodpecker', 'Eastern Bluebird',
               'Eastern Kingbird', 'Eastern Meadowlark', 'Eastern Phoebe', 'Eastern Towhee', 'Eastern Wood Pewee',
               'Eurasian Collared Dove', 'Eurasian Teal', 'European Herring Gull', 'Evening Grosbeak', 'Field Sparrow',
               'Fish Crow', 'Gadwall', 'Golden Eagle', 'Golden-crowned Kinglet', 'Golden-crowned Sparrow',
               'Great Blue Heron', 'Great Crested Flycatcher', 'Great Egret', 'Great Horned Owl',
               'Great-tailed Grackle',
               'Greater Roadrunner', 'Greater Yellowlegs', 'Green Heron', 'Green-tailed Towhee', 'Grey Catbird',
               'Hairy Woodpecker', "Hammond's Flycatcher", 'Hermit Thrush', 'Hooded Warbler', 'Horned Grebe',
               'Horned Lark', 'House Finch', 'House Sparrow', 'House Wren', 'Indigo Bunting', 'Juniper Titmouse',
               'Killdeer', 'Ladder-backed Woodpecker', 'Lark Sparrow', 'Lazuli Bunting', 'Least Bittern',
               'Least Flycatcher', 'Least Sandpiper', 'Lesser Goldfinch', 'Lesser Nighthawk', 'Lesser Yellowlegs',
               "Lincoln's Sparrow", 'Loggerhead Shrike', 'Long-billed Curlew', 'Long-billed Dowitcher',
               'Long-tailed Duck', 'Louisiana Waterthrush', "MacGillivray's Warbler", 'Magnolia Warbler', 'Mallard',
               'Mangrove Warbler', 'Marsh Wren', 'Merlin', 'Mountain Chickadee', 'Mourning Dove', 'Myrtle Warbler',
               'Northern Cardinal', 'Northern Flicker', 'Northern Mockingbird', 'Northern Parula', 'Northern Pintail',
               'Northern Raven', 'Northern Shoveler', 'Northern Waterthrush', 'Olive-sided Flycatcher',
               'Orange-crowned Warbler', 'Ovenbird', 'Pacific-slope Flycatcher', 'Pectoral Sandpiper',
               'Peregrine Falcon', 'Phainopepla', 'Pied-billed Grebe', 'Pileated Woodpecker', 'Pine Grosbeak',
               'Pine Siskin', 'Pine Warbler', 'Pinyon Jay', 'Plumbeous Vireo', 'Prairie Warbler', 'Purple Finch',
               'Pygmy Nuthatch', 'Red Crossbill', 'Red Fox Sparrow', 'Red-bellied Woodpecker', 'Red-breasted Nuthatch',
               'Red-eyed Vireo', 'Red-necked Phalarope', 'Red-shouldered Hawk', 'Red-tailed Hawk',
               'Red-winged Blackbird', 'Ring-billed Gull', 'Rock Dove', 'Rock Wren', 'Rose-breasted Grosbeak',
               'Ruby-crowned Kinglet', 'Rufous Hummingbird', 'Rusty Blackbird', 'Sand Martin', 'Savannah Sparrow',
               "Say's Phoebe", 'Scarlet Tanager', "Scott's Oriole", 'Semipalmated Plover', 'Semipalmated Sandpiper',
               'Short-eared Owl', 'Snow Bunting', 'Snow Goose', 'Solitary Sandpiper', 'Song Sparrow', 'Sora',
               'Spotted Sandpiper', 'Spotted Towhee', "Steller's Jay", "Swainson's Thrush", 'Swamp Sparrow',
               'Tree Swallow', 'Tufted Titmouse', 'Tundra Swan', 'Veery', 'Vesper Sparrow', 'Violet-green Swallow',
               'Warbling Vireo', 'Western Grebe', 'Western Kingbird', 'Western Meadowlark', 'Western Osprey',
               'Western Tanager', 'Western Wood Pewee', 'White-breasted Nuthatch', 'White-crowned Sparrow',
               'White-throated Sparrow', 'Wild Turkey', 'Willow Flycatcher', "Wilson's Snipe", "Wilson's Warbler",
               'Winter Wren', 'Wood Duck', 'Wood Thrush', "Woodhouse's Scrub Jay", 'Yellow-bellied Flycatcher',
               'Yellow-bellied Sapsucker', 'Yellow-headed Blackbird', 'Yellow-throated Vireo']

path = '/Users/allis/Bird Recognition Project/sample_audio/'


def identify():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.subheader("Choose an audio file that you recorded")
    uploaded_audio = st.file_uploader("Select file:")
    if uploaded_audio is not None:
        read_audio = uploaded_audio.read()
        st.audio(read_audio, format='audio/wav')
        sound = AudioSegment.from_mp3(path + uploaded_audio.name)
        sound.export(path + uploaded_audio.name[:-4] + '.wav', format="wav")
        wav_file = path + uploaded_audio.name[:-4] + '.wav'
        pred_feature = features_extract(wav_file)
        pred_input = np.expand_dims(pred_feature, axis=0)
        pred = model.predict(pred_input)
        pred = class_names[np.argmax(pred)]
        st.header("The sound belongs to the species: ")
        st.subheader(pred)
        st.header("")
        audio, sr = librosa.load(wav_file)
        st.caption("Audio waveform:")
        st.pyplot(plot_wavefrom(audio))
        st.caption("")
        st.caption("Audio spectrogram:")
        st.pyplot(plot_spectrogram(audio))


def plot_wavefrom(audio):
    plt.figure(figsize=(10, 5))
    librosa.display.waveshow(audio, x_axis="time")

    return plt.gcf()


def plot_spectrogram(audio):
    stft = librosa.stft(audio)
    audio_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(audio_db, x_axis="time", y_axis="hz", cmap='magma')

    return plt.gcf()


def features_extract(audio_path):
    sr = 22050
    time = 10
    # Loads audio with librosa from audio_path
    audio, _ = librosa.load(audio_path, sr = sr, duration = time, res_type='kaiser_fast')
    # Pads the array to a shape that is suitable for the CNN
    audio = np.pad(audio, (0, sr * time - audio.shape[0]), mode='constant')
    # Extracts the features from the mel spectrogram of each audio file
    spectrogram = librosa.feature.melspectrogram(audio.astype(np.float32),
                                                 sr = sr,
                                                 n_fft = 1024,
                                                 hop_length = 1024,
                                                 n_mels = 128,
                                                 htk = True,
                                                 fmin = 1400,
                                                 fmax = sr / 2)
    # Returns the audio file name and features array
    return spectrogram


if __name__ == "__main__":
    main()
