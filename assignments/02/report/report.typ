#import "template.typ": *
#show: ieee.with(
  title: "Seq2Seq model for machine translation",
  // abstract: [],
  authors: (
    (
      name: "Adam Barla",
      department: [CE7455: Assignment 2],
      // organization: [Affiliation],
      location: [NTU, Singapore],
      email: "n2308836j@e.ntu.edu.sg"
    ),
  ),
  // index-terms: ("A", "B", "C", "D"),
  // bibliography-file: "refs.bib",
)


rouge works better on corpus
repo is trash
bla


#figure(
    [
        #show table.cell.where(x: 10): set text(
          weight: "bold",
        )
        #let t_csv = csv("figures/test_fmeasure.csv")
        #let t = t_csv.map(m => {
        // if v contains * then bold
        m.map(v => {
            if v.contains("*") {
                return [*#v.replace("*", "")*]
            } else {
                return v
            }
        })
        })
        #table(
            stroke:  none,
            columns: (85pt,) +  (t.first().len()-1) * (1fr,),
            align: (horizon, center),
            table.hline(start: 0,stroke:1pt),
            table.header(
            table.cell(rowspan:2,[*Configuration*]), table.cell(colspan: 4, [*Rouge F-Measure*]),
            table.hline(start: 0,stroke:0.5pt),
            [1], [2], [L], [L-Sum],
            ),
            table.hline(start: 0),
            //table.vline(x: 1, start: 1),
            //table.vline(x: 10, start: 1),
            ..t.flatten(),
            table.hline(start: 0,stroke:1pt),
        )
    ],
    caption: [Test F-Measures of the different configurations of the model. The best results are highlighted in bold.],
) <test_fmeasure>



#figure(
    [
        #show table.cell.where(x: 10): set text(
          weight: "bold",
        )
        #let t_csv = csv("figures/test_recall.csv")
        #let t = t_csv.map(m => {
        // if v contains * then bold
        m.map(v => {
            if v.contains("*") {
                return [*#v.replace("*", "")*]
            } else {
                return v
            }
        })
        })
        #table(
            stroke:  none,
            columns: (85pt,) +  (t.first().len()-1) * (1fr,),
            align: (horizon, center),
            table.hline(start: 0,stroke:1pt),
            table.header(
            table.cell(rowspan:2,[*Configuration*]), table.cell(colspan: 4, [*Rouge Recall*]),
            table.hline(start: 0,stroke:0.5pt),
            [1], [2], [L], [L-Sum],
            ),
            table.hline(start: 0),
            //table.vline(x: 1, start: 1),
            //table.vline(x: 10, start: 1),
            ..t.flatten(),
            table.hline(start: 0,stroke:1pt),
        )
    ],
    caption: [Test Recall of the different configurations of the model. The best results are highlighted in bold.],
) <test_recall>

#figure(
    [
        #show table.cell.where(x: 10): set text(
          weight: "bold",
        )
        #let t_csv = csv("figures/test_precision.csv")
        #let t = t_csv.map(m => {
        // if v contains * then bold
        m.map(v => {
            if v.contains("*") {
                return [*#v.replace("*", "")*]
            } else {
                return v
            }
        })
        })
        #table(
            stroke:  none,
            columns: (85pt,) +  (t.first().len()-1) * (1fr,),
            align: (horizon, center),
            table.hline(start: 0,stroke:1pt),
            table.header(
            table.cell(rowspan:2,[*Configuration*]), table.cell(colspan: 4, [*Rouge Precision*]),
            table.hline(start: 0,stroke:0.5pt),
            [1], [2], [L], [L-Sum],
            ),
            table.hline(start: 0),
            //table.vline(x: 1, start: 1),
            //table.vline(x: 10, start: 1),
            ..t.flatten(),
            table.hline(start: 0,stroke:1pt),
        )
    ],
    caption: [Test Precision of the different configurations of the model. The best results are highlighted in bold.],
) <test_recall>
