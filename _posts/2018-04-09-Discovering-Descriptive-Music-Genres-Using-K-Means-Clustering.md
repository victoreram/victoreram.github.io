---
title: Discovering Descriptive Music Genres Using K-Means Clustering
categories:
- Data Science
---
![](https://cdn-images-1.medium.com/max/720/1*Lxw5ngfLacxz5YQ1ar-IBA.png)

I was mulling over what to watch on Netflix one night when it recommended me
“Critically Acclaimed, Visually Striking Crime Dramas”. If Netflix can generate
eerily descriptive movie ‘genres’, why not extend this to music? Just like
movies, we have more ways to describe music than we have existing genres.

So, I sought to find similarities between music using unsupervised machine
learning methods. I chose this route instead of genre classification because
music genre classification is bounded by a wide range of subjectivity. Band A
may be labeled Metal by someone, and Rock by another. Say I was specifically in
the mood for “[Psychedelic Atmospheric Black
Metal](https://www.youtube.com/watch?v=mqK6s93wqRU)”, “[Progressive Thrash Metal
About Sci-Fi](https://www.youtube.com/watch?v=_TanRJlb3NI)”, or “Folk-inspired
Melodeath With Black Metal Influences”. It would be difficult to discover unique
music that would satisfy my mood if I limited myself by speaking the language
within conventional genre labels.

By performing clustering, we can cross the boundaries imposed by genre
classification, findings similarities among music instead of being bound by the
subjectivity of genres. We aren’t necessarily rewriting music genres, as music
genres have cultural and historical nuances that even the most robust machine
learning algorithms can’t strip down. Rather, clustering based on audio features
augment music genres and can derive more descriptive subgenres based on the
audio features from a track.

### Definition of Audio Features

To better understand what the genres are clustered by, audio features must first
be defined. I used the [audio
features](https://developer.spotify.com/web-api/get-audio-features/) defined by
@Spotify:

> acousticness: A confidence measure from 0.0 to 1.0 of whether the track is
> acoustic. 1.0 represents high confidence the track is acoustic.

> danceability: Danceability describes how suitable a track is for dancing based
> on a combination of musical elements including tempo, rhythm stability, beat
strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is
most danceable.

> Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of
> intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.
For example, death metal has high energy, while a Bach prelude scores low on the
scale. Perceptual features contributing to this attribute include dynamic range,
perceived loudness, timbre, onset rate, and general entropy.

> instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah”
> sounds are treated as instrumental in this context. Rap or spoken word tracks
are clearly “vocal”. The closer the instrumentalness value is to 1.0, the
greater likelihood the track contains no vocal content. Values above 0.5 are
intended to represent instrumental tracks, but confidence is higher as the value
approaches 1.0.

> liveness: Detects the presence of an audience in the recording. Higher liveness
> values represent an increased probability that the track was performed live. A
value above 0.8 provides strong likelihood that the track is live.

> speechiness: Speechiness detects the presence of spoken words in a track. The
> more exclusively speech-like the recording (e.g. talk show, audio book, poetry),
the closer to 1.0 the attribute value. Values above 0.66 describe tracks that
are probably made entirely of spoken words. Values between 0.33 and 0.66
describe tracks that may contain both music and speech, either in sections or
layered, including such cases as rap music. Values below 0.33 most likely
represent music and other non-speech-like tracks.

> tempo: The overall estimated tempo of a track in beats per minute (BPM). In
> musical terminology, tempo is the speed or pace of a given piece and derives
directly from the average beat duration.*

> valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed
> by a track. Tracks with high valence sound more positive (e.g. happy, cheerful,
euphoric), while tracks with low valence sound more negative (e.g. sad,
depressed, angry).

*I scaled tempo to also be between 0.0–1.0, and normalized all values for my
feature vector.

### Applying K-Means Clustering On All Tracks

In short, [K-Means
Clustering](https://www.quora.com/What-is-the-k-Means-algorithm-and-how-does-it-work)
is a technique that categorizes data based on the mean characteristics of each
data point. I first absorbed the more obscure genres into the larger ones. (e.g.
Folk has a lot of songs but Blues doesn’t. Blues is closest to Folk by distance,
so I merged it into Folk/Blues). After applying K-Means clustering, I plotted a
heatmap of the audio feature values of each cluster centroid.

<span class="figcaption_hack">A heatmap of audio feature values by K-Means label.</span>

Given the cluster centroids from K-Means, we can see the values which
characterize each K-Means Label. Based on the values from the heatmap, I made
best-guess interpretations of each label in quotes, meant to resemble Netflix’s
disturbingly specific genres. Someone with a wider music vocabulary can easily
think up of better genre names.

* KM0: Highly acoustic and instrumental. Low danceability, energy tempo, valence.
“Slow & Somber Acoustics”
* KM1: Highly instrumental and valent. Mid-tempo, mid-energy. Low acousticness and
speechiness. “Happy & Danceable Instrumentals”
* KM2: Highly instrumental. Low valence, speechiness. “Sad Instrumentals”
* KM3: Highly valent. Speechy. Low instrumentalness. “Upbeat Songs With Cheerful
Vocals”
* KM4: Highly instrumental, danceable, fast. Low acousticness. “Fast & Danceable
Instrumentals”.
* KM5: High energy, valent, and fast. Relatively high liveness. Low acousticness
and instrumentalness. ‘Fast, Upbeat & Cheerful”
* KM6: Highly acoustic. Mid-high danceable. Speechy. Low energy. “Slow Dance”.
* KM7: Highly valent and instrumental. Low tempo and speechiness. “Happy & Slow”
* KM8: High energy, tempo and instrumentalness. Low acousticness and speechiness.
“Happy & Upbeat Instrumentals”

To visualize these features, I applied Principal Component Analysis to reduce
the dataset to 2 dimensions. The feature axis is an estimated visual guide based
on the magnitude and direction of [explained
variance](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues).

<span class="figcaption_hack">A side-by-side comparison between tracks clustered by conventional music genre
(left) and clusters generated by K-Means (right).</span>

Heatmap of audio feature values by K-Means Label

* The left plot with conventional genres showed some structured clusters but is
quite messy overall. Folk/Blues, Classical and Old-Time clustered together
towards strong acoustic values and weak energy values. Metal seemed to straddle
along the instrumentalness axis but skewed towards higher energy values.

The right plot with K-Means labels visibly showed more structure than its
conventional counterpart. The K-Means labels do reasonably describe some of the
genres observed. Classical is sensibly paired with “Slow & Somber Acoustics”.
Folk/Blues is split between “Slow & Somber Acoustics” and “Happy & Danceable
Instrumentals”. Another is Metal — “Happy & Upbeat Instrumentals” straddling
along the top left edge, which may suggest an unfortunate amount of Power Metal
in this dataset.

We can see how these clusters relate to each genre from the following cell:



By matching some of the instances each label, we can confirm some of the
observations on the plot above and see some that are hidden. K-Means reasonably
labeled plenty of Electronic, Hip-Hop, Pop, and Country as “Upbeat Songs With
Cheerful Vocals”. The bottom cell accurately suggests that Electronic music
dominates the variants of “Danceable Instrumentals”.

### Applying K-Means Clustering On Rock Tracks

Another case study I explored was the division between subgenres within a genre:
Rock. I chose K = 5 subgenres: Pop, Indie-Rock, Psychedelic Rock, Punk, and the
remaining “Plain” Rock tracks that don’t fall under those subgenres.

Just as before, I show off my embarrassingly uncreative best-guess
interpretations:

<span class="figcaption_hack">Heatmap of audio feature values by K-Means Label</span>

* KM0: High energy, valence, tempo, danceablity. Low acousticness,
instrumentalness. I’ll attribute the low speechiness and low instrumentalness to
synths. “Upbeat Rock with Synths to Dance to”
* KM1: High acousticness and instrumentalness. Not danceable. Low speechiness,
tempo, and valence. “Slow & Depressing Rock”
* KM2: High acousticness, instrumentalness and valence. Lowest speechiness and
tempo. “Slow & Cheerful Rock”
* KM3: High acousticness and danceability. Low energy. “Slow Dance Rock”
* KM4: High instrumentalness, tempo and energy. Low acousticness, danceability.
“Fast & Energetic Rock”

Applying PCA once again to visualize this, we get the following plots:

There is some logical subgenre clustering to point out, such as Psych-Rock
straddling along the lower right / low valence edge. There’s a lot of Punk
bordering the high energy edge on the bottom left. Though Pop can be seen on the
upper left / high valence and lower right / low valence edge, there is a
somewhat dense Pop cluster on the left most nose of the plot.

The right plot labeled by K-Means cluster labels confirm some intuitive
pairings: Psych-Rock with “Slow & Depressing Rock”, some Indie-Rock basically
being Psych-Rock but more cheerful, Punk being split among “Fast & Energetic
Rock” and “Upbeat Rock With Synths To Dance To” (Wait… Punk has synths these
days??). We examine the genre pairings as before.

    Indie-Rock    Slow & Cheerful Rock
    Pop          Fast & Energetic Rock
    Psych-Rock  Slow & Depressing Rock
    Punk         Fast & Energetic Rock
    Rock         Fast & Energetic Rock

Psych-Rock being mostly paired with “Slow & Depressing Rock” and Punk mostly
paired with “Fast & Energetic Rock” confirms what we see on the scatter plot as
well as how those genres are typically described. “Slow & Cheerful Rock” can be
a useful description for Indie given that we’re comparing them to the extremes
from other Rock subgenres.

### Conclusion

**Describing Music In A New Way**

Clustering music into genres based on audio features allow music to be described
in new ways. Combining these genres with the conventions already employed by
human-labeled genres, new and more descriptive genres can be generated and
labeled onto music.

**Cross-Genre Similarities**

Although the generated genres allow music to be viewed in a different
perspective, the generated music genres were found to be compatible with
human-labeled genres. For example, a Pop fan is likely to enjoy “Upbeat Songs
With Cheerful Vocals” that they wouldn’t otherwise discover if they stuck to
conventional genres.

**Breaking Down Subgenres Further**

This experiment was done with a contained set of tracks and can be scaled to
implement more tracks or even add new audio features (Distortion, Percussion,
etc.). With these future additions in mind, I may finally discover tracks in the
ever elusive genre of Folk-inspired Melodeath With Black Metal Influences.

### References:

[1] [Project
Repo](https://github.com/victoreram/Springboard-Data-Science/tree/master/GenreClustering/).
Contains code and a report that explains my methods.

I used the data from the Free Music Archive (FMA), which has tons more features
and audio data then I could think to play with.

[2] [FMA Github](https://github.com/mdeff/fma)

[3] [FMA: A Dataset for Music Analysis](https://arxiv.org/abs/1612.01840)

__________________________________________________________________

### LatinX in AI Coalition Mission:

#### Creating Harmony Between AI and the Latinx Community

* Increase representation of Latinx in Artificial Intelligence
* Improve access to education and resources in AI engineering to latinx community
* Improve awareness of the long and short term effects of artificial intelligence
technology on the Latinx community
* Increase communication between AI companies, engineers, researchers and the
Latinx community
* Ensure transparency and accuracy of latinx culture and voice in data
representation

**Do you identify as latinx and are working in artificial intelligence or know
someone who is latinx and is working in artificial intelligence?**

Add to our directory:
[https://bit.ly/LatinXinAI-Directory-Form](https://bit.ly/LatinXinAI-Directory-Form)

**Check out our open source website:**
[https://www.latinxinai.org/](https://www.latinxinai.org/)

#### If you enjoyed reading this, you can contribute good vibes (and help more people
discover this post and our community) by hitting the 👏 below — it means a lot!

* [Music](https://medium.com/tag/music?source=post)
* [Data Science](https://medium.com/tag/data-science?source=post)
* [Machine Learning](https://medium.com/tag/machine-learning?source=post)
* [Artificial
Intelligence](https://medium.com/tag/artificial-intelligence?source=post)
* [AI](https://medium.com/tag/ai?source=post)

From a quick cheer to a standing ovation, clap to show how much you enjoyed this
story.

### [Victor Ramirez](https://medium.com/@victor.em.ramirez)

### [LatinxInAI](https://medium.com/latinxinai?source=footer_card)

Latinx in AI Coalition - Creating Harmony Between AI and the Latinx Community
