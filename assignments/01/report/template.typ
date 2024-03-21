// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let project(title: "", abstract: [], authors: (), body) = {
  // Set the document's basic properties.
  set document(author: authors.map(a => a.name), title: title)
  set page(numbering: "1", number-align: center, margin: 1cm)
  set text(font: "Linux Libertine", lang: "en", size: 10pt)

  show link: set text(blue)
  show link: underline
  show figure: set block(breakable: true)

  // Set paragraph spacing.
  show par: set block(above: 0.75em, below: 0.75em)

  set heading(numbering: "1.1.1")

  // Set run-in subheadings, starting at level 3.
  show heading: it => {
    if it.level > 3 {
      parbreak()
      text(11pt, style: "italic", weight: "regular", it.body + ".")
    } else {
      it
    }
  }

  set par(leading: 0.58em)

  // Title row.
  align(center)[
    #block(text(weight: 700, 1.75em, title))
  ]

  // Author information.
  pad(
    top: 0.3em,
    bottom: 0.3em,
    x: 2em,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(center)[
        *#author.name* \
        #author.email \
        #author.affiliation
      ]),
    ),
  )

//  Main body
      
  set par(justify: true)
  show: columns.with(2, gutter: 1em)

  body
}