#import "@preview/simple-research-poster:0.2.0": *
#import "colors.typ": (
  base-colors,
  bold-color,
)
#import "sections.typ": *

#set page(
  paper:   "a1",
  flipped: true,
  margin:  0%,
)

#show: poster.with(
  title:       text(size: 54pt, weight: "extrabold")[Reinforcement Learning for Autonomous Rover Exploration in Unknown Environments],
  author:      text(size: 40pt)[James Ogle --- Professor: Fei Liu],
  subtitle:    text(size: 28pt)[University of Tennessee Knoxville],
  logo:        image("assets/UTSystem-LogoKit/UTSystem_CenterAlign/UTSystem_Center_WEB/PNG/UTSystem_Center_RGB-white_trns.png", height: 120%),
  base-colors: base-colors,
)
#let colored-poster-section = poster-section.with(
  base-colors: base-colors,
  title-style: text.with(
    size:   40pt,
    weight: "extrabold",
    fill:   base-colors.bgcolor2
  ),
)

#set text(
  size: 25pt,
)
#set par(justify: true)  /* Makes it where all text at paragraph ends are lined up */
#show strong: set text(fill: bold-color)

#pad(
  grid(
    columns: 3,
    inset:   35pt,
    gutter:  30pt,
    [
      #colored-poster-section[Introduction][#section1]
      #colored-poster-section(fill: true)[Method/Design][#section2]
    ],
    [
      #colored-poster-section[Results][#section3]
    ],
    [
      #colored-poster-section(fill: true)[Conclusion][#section6]
      #colored-poster-section[Acknowledgements][#acknowledgements]
      #colored-poster-section(fill: true)[References][#references]
    ]
  ),
  top: 20pt,
  x: 70pt,
)
