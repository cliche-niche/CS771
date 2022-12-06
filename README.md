This repository contains my lecture notes and assignment submissions made in the course [CS771](https://web.cse.iitk.ac.in/users/purushot/courses/ml/2022-23-a/) (2022-23 Sem. I) in a team of 5. Our team name was `ML Class of '14` with the team members:

|Name|User ID|
|:-:|:-:|
|Kunwar Preet Singh|[Enkryp](https://github.com/Enkryp)|
|Parinay Chauhan|[parinayc20](https://github.com/parinayc20)|
|Pratyush Gupta|[PratyushGupta0](https://github.com/PratyushGupta0)|
|Akhil Agrawal|[akhilagrawal1001](https://github.com/akhilagrawal1001)|
|Aditya Tanwar (me)|[cliche-niche](https://github.com/cliche-niche)|
---

<br>

## Notes
A major component of my notes were made from the slides of discussion sessions taken by Prof. Purushottam Kar, along with the videos uploaded on [the course's](https://www.youtube.com/channel/UC02HzhBF8Wg2zXhvWebz1tA) YouTube channel. The slides are available on the [course website](https://web.cse.iitk.ac.in/users/purushot/courses/ml/2022-23-a/) [IIT-K intranet/ VPN required]. <br>
The notes had to be split into two parts due to size constraints of files. The [first half](./Notes/CS771-1.pdf) almost corresponds to content covered upto and including the 7<sup>th</sup> discussion session, while the [second half](./Notes/CS771-2.pdf) contains the rest.

## Assignments
In each assignment, we were given a problem statement after being taught some methods/models:

+ [<u>Assignment 1:</u>](./assn1/) Exercise on using [SVMs](https://en.wikipedia.org/wiki/Support-vector_machine). We explored SGDM to solve it, but ended up submitting SCDM with a bunch of case-specific optimizations.
+ [<u>Assignment 2:</u>](./assn2/) The problem was that of multilabel classification, with complete freedom to use whatever one wants. Our group ended up using a mix of [Decision Trees](https://en.wikipedia.org/wiki/Decision_tree), and [Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network).
+ [<u>Assignment 3:</u>](./assn3/) The problem required the usage of [Computer Vision](https://en.wikipedia.org/wiki/Computer_vision) followed by [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning), specifically [CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network), to infer greek alphabets written in a CAPTCHA. A perfectly arbitrary captcha has been provided [here](./assn3/train/12.png).

Note: We were provided with 2000 sample images for [A3](./assn3/) but to save space on this repository, a single image has been provided.

### Marks

+ [A1:](./assn1/) 66/60 - 7 marks bonus for using reduced dimensionality (`<200`) and 1 mark deducted for `t=0.2s` and `0.01<=e=0.04<1` (full marks for `e<0.01`)
+ [A2:](./assn2/) 87/100 - 6 marks deducted for a size of `~7.5 MB` (full marks for `<128 KB`), 6 marks deducted for an inference time of `~2s` (`ceil( max( 1 - t, 0.4 ) * 10 )`), 1 mark deducted for `mprec@1 = 0.625` (`ceil( max( p, 0.4 ) * 5 )`)
+ [A3:](./assn3/) 89/100 - 6 marks deducted for a size of `~11.6 KB` (full marks for `<128 KB`), 4 marks deducted for an inference time of `~30s` (full marks for `<5s`), 1 mark deducted for inaccuracy (`floor( c * 40 )`)

<br>

## Acknowledgement
+ The course website can be found [here](https://web.cse.iitk.ac.in/users/purushot/courses/ml/2022-23-a/) [Intranet required].
+ This repository has been forked from [Kunwar's repository](https://github.com/Enkryp/CS771). Naturally, there are a lot of similarities between the two, except for the [readme](./README.md) and the [notes](./Notes/) section.
+ [Kunwar Preet "Enkryp" Singh](https://github.com/Enkryp) for Assignments 1 and 2 (and the bonus 7 marks!)
+ [Pratyush "PratyushGupta0" Gupta](https://github.com/PratyushGupta0) for Assignment 3.