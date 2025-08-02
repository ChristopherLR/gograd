package grad

import (
	"fmt"
	"os"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
)

func GenGraph() {
	tr, _, _ := GenDataYinYang(random, 10000, 0.1, 0.5)

	sc := charts.NewScatter()
	sc.SetGlobalOptions(
		charts.WithTitleOpts(opts.Title{Title: "Yin–Yang data"}),
		charts.WithXAxisOpts(opts.XAxis{Name: "x"}),
		charts.WithYAxisOpts(opts.YAxis{Name: "y"}),
	)

	series := map[uint8][]opts.ScatterData{}
	for _, p := range tr {
		series[p.C] = append(series[p.C], opts.ScatterData{Name: "", Value: []float64{p.X, p.Y}})
	}

	sc.AddSeries("yin", series[0])
	sc.AddSeries("yang", series[1])
	sc.AddSeries("dots", series[2])

	f, _ := os.Create("yinyang.html")
	err := sc.Render(f)
	if err != nil {
		fmt.Println(err)
	}
}
