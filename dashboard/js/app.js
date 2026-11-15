
/* Configuration */

const el = id => document.getElementById(id);

const regionLabels = {
    metro_manila_approximate: 'Metro Manila',
    benguet_approximate: 'Benguet',
    angat_vicinity: 'Angat vicinity',
    philippines_bbox: 'Philippines (bounding box)'
};

const indicatorNames = {
    GPM_3IMERGDF_07_precipitation: 'Daily precipitation (IMERG)',
    rainfall_mm_month: 'Monthly rainfall (IMERG)',
    M2SMNXEDI_2_R20mm: 'Days with ≥20 mm rainfall',
    M2SMNXEDI_2_R95D: 'Days above 95th percentile',
    M2SMNXEDI_2_RX1Day: 'Maximum 1-day rainfall',
    M2SMNXEDI_2_RX5Day: 'Maximum 5-day rainfall',
    M2SMNXEDI_2_RX5Daycount: 'Heavy 5-day rainfall periods',
    M2SMNXEDI_2_CDD: 'Longest dry spell',
    M2SMNXEDI_2_drydays: 'Dry days',
    M2SMNXEDI_2_wetdays: 'Wet days',
    M2TMNXLND_5_12_4_GWETROOT: 'Root-zone wetness',
    M2TMNXLND_5_12_4_RZMC: 'Root-zone soil moisture',
    M2TMNXLND_5_12_4_SFMC: 'Surface soil moisture',
    M2SMNXSLV_5_12_4_T2MMAX: 'Maximum daily air temperature',
    M2SMNXSLV_5_12_4_T2MMEAN: 'Mean air temperature',
    M2SMNXSLV_5_12_4_T2MMIN: 'Minimum daily air temperature'
};

const state = {
    mode: 'interpret',
    chartView: 'trend',
    region: 'metro_manila_approximate',
    datasetId: null,
    datasetId2: null,
    compare: false,
    period: 'all',
    question: '',
    result: null
};

let catalog = [];
let renderToken = 0;
let aiRequestToken = 0;
let predictionToken = 0;
let correlationToken = 0;
let correlationResult = null;

const predictionCache = new Map();


/* Utilities */

function pretty(value, decimals = 2) {
    if (value === null || value === undefined) {
        return '—';
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return '—';
    }

    return number.toLocaleString('en-US', {
        maximumFractionDigits: decimals
    });
}

function label(dataset) {
    return indicatorNames[dataset.variable] || dataset.variable;
}

function analyzedUnit(data) {
    const unit = data.unit || 'unknown';

    if (data.aggregation === 'sum') {
        const conversions = {
            'days/month': 'days/year',
            'mm/month': 'mm/year',
            'periods/month': 'periods/year'
        };

        return conversions[unit] || unit;
    }

    return unit;
}

function setLoading(value) {
    el('loadStatus').hidden = !value;

    el('dashboardGrid').classList.toggle(
        'is-loading',
        value
    );
}

function setError(message) {
    el('trendChart').textContent = message;
    el('aiContent').textContent = message;

    [
        'meanMetric',
        'maxMetric',
        'trendMetric'
    ].forEach(id => {
        el(id).textContent = '—';
    });
}

function addInsightCard(container, title, text) {
    const card = document.createElement('div');
    card.className = 'insight-card';

    const heading = document.createElement('div');
    heading.className = 'insight-title';
    heading.textContent = title;

    const body = document.createElement('p');
    body.textContent = text;
    body.style.whiteSpace = 'pre-line';

    card.append(heading, body);
    container.append(card);

    return card;
}


/* Dataset Selection */

function refreshIndicators() {
    const entries = catalog.filter(dataset => {
        return dataset.region === state.region;
    });

    const select = el('indicator');

    select.replaceChildren(
        ...entries.map(dataset => {
            return new Option(
                label(dataset),
                dataset.path
            );
        })
    );

    if (!entries.some(dataset => {
        return dataset.path === state.datasetId;
    })) {
        state.datasetId = entries[0]?.path || null;
    }

    select.value = state.datasetId || '';

    refreshSecondIndicator();
}

function refreshSecondIndicator() {
    const entries = catalog.filter(dataset => {
        return (
            dataset.region === state.region &&
            dataset.path !== state.datasetId
        );
    });

    const select = el('indicator2');

    select.replaceChildren(
        ...entries.map(dataset => {
            return new Option(
                label(dataset),
                dataset.path
            );
        })
    );

    if (!entries.some(dataset => {
        return dataset.path === state.datasetId2;
    })) {
        state.datasetId2 = entries[0]?.path || null;
    }

    select.value = state.datasetId2 || '';
    select.hidden = !state.compare;

    const button = el('toggleIndicator');

    button.textContent = state.compare ? '−' : '+';

    button.title = state.compare
        ? 'Remove comparison variable'
        : 'Add comparison variable';

    button.setAttribute(
        'aria-label',
        button.title
    );
}

function toggleComparison() {
    state.compare = !state.compare;

    if (!state.compare) {
        state.datasetId2 = null;
    }

    refreshSecondIndicator();
    updateDashboard();
}


/* Historical Chart */

function renderResult(data, data2 = null) {
    state.result = data;

    const dataset = catalog.find(item => {
        return item.path === data.dataset_id;
    });

    const dataset2 = data2
        ? catalog.find(item => {
            return item.path === data2.dataset_id;
        })
        : null;

    const name1 = dataset ? label(dataset) : data.variable;
    const name2 = dataset2 ? label(dataset2) : data2?.variable;

    const unit1 = analyzedUnit(data);
    const unit2 = data2 ? analyzedUnit(data2) : null;

    const series = data.series || [];

    const x = series.map(item => item.year);
    const y = series.map(item => item.value);

    el('chartTitle').textContent = data2
        ? `${name1} vs ${name2}`
        : name1;

    el('chartSubtitle').textContent =
        `${regionLabels[data.region] || data.region} · NASA observations`;

    el('sourceLabel').textContent =
        `NASA ${data.source_format === 'giovanni'
            ? 'GIOVANNI'
            : 'EARTH ENGINE'}`;

    el('rangeLabel').textContent = series.length
        ? `${series[0].year} — ${series[series.length - 1].year}`
        : 'No observations';

    el('meanMetric').textContent = pretty(data.summary?.mean);
    el('maxMetric').textContent = pretty(data.summary?.maximum);

    el('maxYear').textContent =
        `Peak in ${data.summary?.max_year ?? '—'}`;

    el('meanUnit').textContent = unit1;
    el('maxUnit').textContent = unit1;

    const slope = data.statistics?.slope_per_decade;

    el('trendMetric').textContent = Number.isFinite(slope)
        ? `${slope >= 0 ? '+' : ''}${pretty(slope)}`
        : '—';

    el('trendUnit').textContent =
        `${unit1} per decade`;

    const traces = [
        {
            x,
            y,
            type: 'scatter',
            mode: 'lines+markers',
            name: name1,
            yaxis: 'y',
            line: {
                color: '#5ce1db',
                width: 2.5
            },
            marker: {
                color: '#5ce1db',
                size: 4
            },
            hovertemplate:
                '%{x}: %{y:.2f}<extra>%{fullData.name}</extra>'
        }
    ];

    if (data.statistics && x.length) {
        const trend = data.statistics;

        traces.push({
            x,
            y: x.map(year => {
                return (
                    trend.intercept +
                    trend.slope_per_year * (year - x[0])
                );
            }),
            type: 'scatter',
            mode: 'lines',
            name: `${name1} trend`,
            yaxis: 'y',
            line: {
                color: '#5ce1db',
                width: 2,
                dash: 'dash'
            },
            hoverinfo: 'skip',
            showlegend: false
        });
    }

    if (data2) {
        const series2 = data2.series || [];

        const x2 = series2.map(item => item.year);
        const y2 = series2.map(item => item.value);

        traces.push({
            x: x2,
            y: y2,
            type: 'scatter',
            mode: 'lines+markers',
            name: name2,
            yaxis: 'y2',
            line: {
                color: '#ffbf69',
                width: 2.5
            },
            marker: {
                color: '#ffbf69',
                size: 4
            },
            hovertemplate:
                '%{x}: %{y:.2f}<extra>%{fullData.name}</extra>'
        });

        if (data2.statistics && x2.length) {
            const trend2 = data2.statistics;

            traces.push({
                x: x2,
                y: x2.map(year => {
                    return (
                        trend2.intercept +
                        trend2.slope_per_year * (year - x2[0])
                    );
                }),
                type: 'scatter',
                mode: 'lines',
                name: `${name2} trend`,
                yaxis: 'y2',
                line: {
                    color: '#ffbf69',
                    width: 2,
                    dash: 'dash'
                },
                hoverinfo: 'skip',
                showlegend: false
            });
        }
    }

    if (typeof Plotly === 'undefined') {
        el('trendChart').textContent =
            'Plotly could not load.';
        return;
    }

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        autosize: true,

        font: {
            color: '#98a8ba',
            family: 'DM Sans, sans-serif',
            size: 11
        },

        margin: {
            t: data2 ? 40 : 15,
            l: 65,
            r: data2 ? 75 : 15,
            b: 45
        },

        showlegend: Boolean(data2),

        legend: {
            orientation: 'h',
            x: 0,
            y: 1.12,
            font: {
                size: 10
            }
        },

        hovermode: 'x unified',

        xaxis: {
            showgrid: false,
            zeroline: false,
            tickfont: {
                color: '#8290a3'
            }
        },

        yaxis: {
            title: {
                text: unit1,
                font: {
                    size: 10,
                    color: '#5ce1db'
                }
            },
            gridcolor: '#273444',
            zeroline: false,
            tickfont: {
                color: '#5ce1db'
            }
        }
    };

    if (data2) {
        layout.yaxis2 = {
            title: {
                text: unit2,
                font: {
                    size: 10,
                    color: '#ffbf69'
                }
            },
            overlaying: 'y',
            side: 'right',
            showgrid: false,
            zeroline: false,
            tickfont: {
                color: '#ffbf69'
            }
        };
    }

    Plotly.react('trendChart', traces, layout, {
        responsive: true,
        displayModeBar: false,
        displaylogo: false
    });
}


/* Monthly Seasonal and Anomaly Views */

async function renderMonthlyView(datasetId, period, view, token) {
    const params = new URLSearchParams({ dataset_id: datasetId, period });
    const response = await fetch(`/api/seasonal?${params}`);
    const body = await response.json();
    if (!response.ok) {
        throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${response.status}`);
    }
    if (token !== renderToken || state.chartView !== view) return;

    const observations = body.observations || [];
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const years = [...new Set(observations.map(o => o.year))].sort((a,b) => a-b);
    const lookup = new Map(observations.map(o => [`${o.year}-${o.month}`, o]));
    const unit = body.unit || '';
    const title = catalog.find(d => d.path === datasetId);
    el('chartTitle').textContent = `${title ? label(title) : body.variable} — ${view === 'seasonal' ? 'Seasonal patterns' : 'Monthly anomalies'}`;
    el('chartSubtitle').textContent = `${regionLabels[body.region] || body.region} · Complete monthly NASA observations`;
    el('rangeLabel').textContent = `${body.start} — ${body.end}`;
    el('seasonalNote').hidden = false;
    el('seasonalNote').textContent = `${body.complete_months} complete months · ${body.incomplete_months} incomplete/missing months · Baseline: same calendar month in selected range · Primary variable only`;

    if (typeof Plotly === 'undefined') {
        el('trendChart').textContent = 'Plotly is unavailable.';
        return;
    }
    const base = {
        paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
        font:{color:'#98a8ba',family:'DM Sans, sans-serif',size:11},
        margin:{t:22,l:65,r:24,b:65}, autosize:true,
        xaxis:{title:'Month',tickfont:{color:'#9aabc0'}},
        yaxis:{title:view === 'seasonal' ? unit : 'Year',tickfont:{color:'#9aabc0'}},
        showlegend:false
    };
    let traces;
    if (view === 'seasonal') {
        traces = [{
            type:'bar', x:months,
            y:body.climatology.map(c => c.mean),
            marker:{color:'#5ce1db'},
            customdata:body.climatology.map(c => c.count),
            hovertemplate:`%{x}: %{y:.3f} ${unit}<br>Complete observations: %{customdata}<extra></extra>`
        }];
        base.yaxis.title = unit;
    } else {
        const z = years.map(y => months.map((m, i) => {
            const obs = lookup.get(`${y}-${i+1}`);
            return obs && obs.anomaly !== null ? obs.anomaly : null;
        }));
        traces = [{
            type:'heatmap', x:months, y:years.map(String), z,
            colorscale:[[0,'#e89b71'],[0.5,'#172536'],[1,'#5ce1db']],
            zmid:0, hoverongaps:false,
            colorbar:{title:unit, tickfont:{color:'#98a8ba'}},
            hovertemplate:`%{y} %{x}: %{z:.3f} ${unit}<extra></extra>`
        }];
        base.yaxis.title = 'Year';
        base.margin.r = 65;
    }
    await Plotly.react('trendChart', traces, base, {responsive:true,displayModeBar:false,displaylogo:false});
}

function setChartView(view) {
    if (!['trend','seasonal','anomalies'].includes(view)) return;
    state.chartView = view;
    updateAIHeading();
    document.querySelectorAll('.chart-view').forEach(button => {
        const active = button.dataset.view === view;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
    });
    updateDashboard();
}

/* Historical Data Requests */

async function updateDashboard() {
    const token = ++renderToken;

    aiRequestToken++;
    predictionToken++;
    correlationToken++;
    correlationResult = null;
    renderCorrelation();
    setLoading(true);

    try {
        if (!state.datasetId) {
            throw new Error('No dataset selected');
        }

        const getDataset = async datasetId => {
            const params = new URLSearchParams({
                dataset_id: datasetId,
                period: state.period
            });

            const response = await fetch(
                `/api/timeseries?${params.toString()}`
            );

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));

                throw new Error(
                    typeof error.detail === 'string'
                        ? error.detail
                        : `HTTP ${response.status}`
                );
            }

            return response.json();
        };

        const requests = [
            getDataset(state.datasetId)
        ];

        if (state.compare && state.datasetId2) {
            requests.push(
                getDataset(state.datasetId2)
            );
        }

        const results = await Promise.all(requests);

        if (token !== renderToken) {
            return;
        }

        // Always update historical metrics, but only show the annual chart in Trend view.
        if (state.chartView === 'trend') {
            el('seasonalNote').hidden = true;
            renderResult(results[0], results[1] || null);
        } else {
            // RenderResult updates the shared metric cards; the monthly view replaces its chart.
            renderResult(results[0], results[1] || null);
            await renderMonthlyView(state.datasetId, state.period, state.chartView, token);
            if (token !== renderToken) return;
        }
        if (state.compare && state.chartView === 'trend') {
       requestCorrelation();
        }
        if (state.mode === 'predict') {
            requestPrediction();
        } else {
            requestAI();
        }

    } catch (error) {
        if (token === renderToken) {
            setError(
                `Unable to load NASA data: ${error.message}`
            );
        }

    } finally {
        if (token === renderToken) {
            setLoading(false);
        }
    }
}


/* Qwen Interpretation */

async function requestAI() {
    const token = ++aiRequestToken;

    if (!state.datasetId || state.mode !== 'interpret') {
        return;
    }

    const datasetId = state.datasetId;
    const period = state.period;
    const view = state.chartView;
    const defaultQuestions = {
    trend: 'Explain the historical annual trend.',
    seasonal: 'Explain the historical seasonal pattern.',
    anomalies: 'Explain the observed monthly anomalies.'
};

const question = state.question ||
    defaultQuestions[view];

    const content = el('aiContent');
    content.textContent = 'Analyzing NASA observations...';

    try {
        const response = await fetch('/api/interpret', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: datasetId,
                period,
                view: state.chartView,
                question
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                typeof result.detail === 'string'
                    ? result.detail
                    : 'Interpretation failed'
            );
        }

        if (
            token !== aiRequestToken ||
            state.mode !== 'interpret' ||
            state.datasetId !== datasetId ||
            state.period !== period ||
            state.chartView !== view
        ) {
            return;
        }

        renderInterpretation(result);

    } catch (error) {
        if (
            token === aiRequestToken &&
            state.mode === 'interpret' &&
            state.datasetId === datasetId &&
            state.period === period &&
             state.chartView === view
        ) {
            content.textContent =
                `Interpretation unavailable: ${error.message}`;
        }
    }
}


/* Qwen Interpretation Display */

function renderInterpretation(result) {
    const content = el('aiContent');
    content.replaceChildren();

    const stats = result.statistics;

    if (result.view === 'trend' && stats) {
        const trend = stats.trend || {};
        const testing = stats.statistical_testing || {};

        const significance =
            testing.significance_status ===
            'preliminary_due_to_autocorrelation'
                ? 'Preliminary — autocorrelation detected'
                : 'Unadjusted statistical test';

        addInsightCard(
            content,
            'VERIFIED STATISTICS',
            `Mean: ${pretty(stats.mean)} ${stats.unit}\n` +
            `Maximum: ${pretty(stats.maximum)} ${stats.unit}\n` +
            `Trend: ${pretty(trend.slope_per_decade)} ` +
            `${trend.slope_unit || ''}\n` +
            `Significance: ${significance}`
        );
    }

    const card = document.createElement('div');
    card.className = 'insight-card';

    const title = document.createElement('div');
    title.className = 'insight-title';
    title.textContent = 'AI INTERPRETATION';

    const explanation = document.createElement('div');
    explanation.className = 'ai-explanation';

    if (
        typeof marked !== 'undefined' &&
        typeof DOMPurify !== 'undefined'
    ) {
        explanation.innerHTML = DOMPurify.sanitize(
            marked.parse(result.answer || '')
        );
    } else {
        explanation.textContent = result.answer || '';
        explanation.style.whiteSpace = 'pre-wrap';
    }

    card.append(title, explanation);
    content.append(card);
    renderCorrelation();
}


/* TimesFM Prediction Request */

async function requestPrediction() {
    const token = ++predictionToken;

    if (!state.datasetId || state.mode !== 'predict') {
        return;
    }

    const datasetId = state.datasetId;
    const horizon = 6;

    const cacheKey = `${datasetId}:monthly:${horizon}`;

    const content = el('aiContent');
    content.replaceChildren();

    const loading = document.createElement('p');
    loading.textContent = 'Generating TimesFM forecast...';
    content.append(loading);

    try {
        let result = predictionCache.get(cacheKey);

        if (!result) {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    dataset_id: datasetId,
                    frequency: 'monthly',
                    horizon
                })
            });

            const body = await response.json();

            if (!response.ok) {
                throw new Error(
                    typeof body.detail === 'string'
                        ? body.detail
                        : 'Prediction failed'
                );
            }

            result = body;
            predictionCache.set(cacheKey, result);
        }

        if (
            token !== predictionToken ||
            state.mode !== 'predict' ||
            state.datasetId !== datasetId
        ) {
            return;
        }

        renderPrediction(result);

    } catch (error) {
        if (
            token === predictionToken &&
            state.mode === 'predict' &&
            state.datasetId === datasetId
        ) {
            content.textContent =
                `Prediction unavailable: ${error.message}`;
        }
    }
}


/* TimesFM Prediction Display */


/* TimesFM Prediction Display */

function renderPrediction(result) {
    const content = el('aiContent');
    content.replaceChildren();

    const history = result.history || [];
    const forecast = result.forecast || [];
    const unit = result.unit || '';

    // Forecast Summary
    addInsightCard(
        content,
        'TIMESFM FORECAST',
        `${result.horizon} ${result.frequency} periods ahead\n` +
        `Historical data through ${result.historical_end}\n` +
        'Experimental model estimates'
    );

    // Forecast Chart
    const chart = document.createElement('div');
    chart.className = 'prediction-chart';
    content.append(chart);

    const observedTrace = {
        x: history.map(item => item.period),
        y: history.map(item => item.value),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'NASA observations',
        line: {
            color: '#5ce1db',
            width: 2.5
        },
        marker: {
            color: '#5ce1db',
            size: 5
        },
        hovertemplate:
            '%{x}: %{y:.2f}<extra>%{fullData.name}</extra>'
    };

    // Connect forecast to last historical observation
    const forecastPeriods = [
        ...(history.length
            ? [history[history.length - 1].period]
            : []),
        ...forecast.map(item => item.period)
    ];

    const forecastValues = [
        ...(history.length
            ? [history[history.length - 1].value]
            : []),
        ...forecast.map(item => item.value)
    ];

    const forecastTrace = {
        x: forecastPeriods,
        y: forecastValues,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'TimesFM forecast',
        line: {
            color: '#ffbf69',
            width: 2.5,
            dash: 'dash'
        },
        marker: {
            color: '#ffbf69',
            size: 5
        },
        hovertemplate:
            '%{x}: %{y:.2f}<extra>%{fullData.name}</extra>'
    };

    // Chart Rendering
    if (typeof Plotly !== 'undefined') {
        const traces = [];

        if (history.length) {
            traces.push(observedTrace);
        }

        if (forecast.length) {
            traces.push(forecastTrace);
        }

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            autosize: true,

            margin: {
                t: 45,
                l: 48,
                r: 12,
                b: 45
            },

            font: {
                color: '#98a8ba',
                family: 'DM Sans, sans-serif',
                size: 10
            },

            showlegend: true,

            legend: {
                orientation: 'h',
                x: 0,
                y: 1.18,
                font: {
                    size: 10
                }
            },

            hovermode: 'x unified',

            xaxis: {
                type: 'category',
                showgrid: false,
                tickangle: -35,
                tickfont: {
                    size: 9
                }
            },

            yaxis: {
                title: {
                    text: unit,
                    font: {
                        size: 10
                    }
                },
                gridcolor: '#273444',
                zeroline: false,
                rangemode: 'tozero'
            },

            shapes: history.length && forecast.length
                ? [{
                    type: 'line',
                    xref: 'x',
                    yref: 'paper',
                    x0: history[history.length - 1].period,
                    x1: history[history.length - 1].period,
                    y0: 0,
                    y1: 1,
                    line: {
                        color: '#8290a3',
                        width: 1,
                        dash: 'dot'
                    }
                }]
                : []
        };

        Plotly.newPlot(
            chart,
            traces,
            layout,
            {
                responsive: true,
                displayModeBar: false,
                displaylogo: false
            }
        );

    } else {
        chart.textContent = 'Plotly is unavailable.';
    }

    // Predicted Values
    if (forecast.length) {
        const forecastText = forecast
            .map(item => {
                return (
                    `${item.period}: ` +
                    `${pretty(item.value)} ${unit}`
                );
            })
            .join('\n');

        addInsightCard(
            content,
            'PREDICTED VALUES',
            forecastText
        );
    }

    // Single Holdout Backtest
    if (result.backtest) {
        const test = result.backtest;

        const improvement = test.baseline_mae > 0
            ? (
                (test.baseline_mae - test.timesfm_mae) /
                test.baseline_mae * 100
            )
            : null;

        const backtestText =
            `TimesFM MAE: ${pretty(test.timesfm_mae, 3)} ${unit}\n` +
            `Baseline MAE: ${pretty(test.baseline_mae, 3)} ${unit}\n` +
            (
                improvement === null
                    ? 'Error reduction unavailable'
                    : `Error reduction: ${pretty(improvement, 1)}%`
            ) +
            '\nSingle historical holdout only.';

        addInsightCard(
            content,
            'HISTORICAL BACKTEST',
            backtestText
        );
    }

    // Rolling Backtest
    if (result.rolling_backtest) {
        const test = result.rolling_backtest;

        const improvement = test.baseline_mae > 0
            ? (
                (test.baseline_mae - test.timesfm_mae) /
                test.baseline_mae * 100
            )
            : null;

        const rollingText =
            `Windows tested: ${test.windows}\n` +
            `Forecast horizon: ${test.horizon} periods\n` +
            `TimesFM MAE: ${pretty(test.timesfm_mae, 3)} ${unit}\n` +
            `Baseline MAE: ${pretty(test.baseline_mae, 3)} ${unit}\n` +
            (
                improvement === null
                    ? 'Error reduction unavailable'
                    : `Error reduction: ${pretty(improvement, 1)}%`
            ) +
            '\nExpanding-window historical evaluation.';

        addInsightCard(
            content,
            'ROLLING BACKTEST',
            rollingText
        );
    }

    // Scientific Limitations
    addInsightCard(
        content,
        'SCIENTIFIC LIMITATIONS',
        (result.warnings || []).join(' ')
    );
}


function updateAIHeading() {
    if (state.mode === 'predict') {
        el('aiHeading').textContent =
            'Future projections';

        el('aiSubheading').textContent =
            'TimesFM experimental forecasting';

        return;
    }

    const headings = {
        trend: {
            title: 'Understand the trend',
            subtitle: 'Annual trend interpretation'
        },
        seasonal: {
            title: 'Understand seasonality',
            subtitle: 'Historical monthly patterns'
        },
        anomalies: {
            title: 'Understand anomalies',
            subtitle: 'Deviations from seasonal averages'
        }
    };

    const selected = headings[state.chartView];

    el('aiHeading').textContent = selected.title;
    el('aiSubheading').textContent = selected.subtitle;
}

/* Mode Switching */

function setMode(mode) {
    state.mode = mode;

    aiRequestToken++;
    predictionToken++;

    document.querySelectorAll('.mode').forEach(button => {
        const active = button.dataset.mode === mode;

        button.classList.toggle('active', active);

        button.setAttribute(
            'aria-pressed',
            String(active)
        );
    });

    el('aiHeading').textContent = mode === 'predict'
        ? 'Future projections'
        : 'Understand the signal';

    el('aiSubheading').textContent = mode === 'predict'
        ? 'TimesFM experimental forecasting'
        : 'NASA statistical interpretation';

    if (mode === 'predict') {
        requestPrediction();
    } else {
        requestAI();
        renderCorrelation();
    }
}


/* Question Submission */

function submitQuestion(event) {
    event.preventDefault();

    const input = el('question');
    const question = input.value.trim();

    if (!question) {
        return;
    }

    if (state.mode === 'predict') {
        input.placeholder =
            'Prediction uses the selected dataset';
        input.value = '';
        return;
    }

    state.question = question;
    input.value = '';

    requestAI();
}


/* Correlation Request */

async function requestCorrelation() {
    const token = ++correlationToken;

    correlationResult = null;
    renderCorrelation();

    if (
        !state.compare ||
        !state.datasetId ||
        !state.datasetId2 ||
        state.chartView !== 'trend'
    ) {
        return;
    }

    const datasetId = state.datasetId;
    const datasetId2 = state.datasetId2;
    const period = state.period;

    try {
        const response = await fetch('/api/correlation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: datasetId,
                dataset_id2: datasetId2,
                period
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                typeof result.detail === 'string'
                    ? result.detail
                    : 'Correlation unavailable'
            );
        }

        if (
            token !== correlationToken ||
            !state.compare ||
            state.chartView !== 'trend' ||
            state.datasetId !== datasetId ||
            state.datasetId2 !== datasetId2 ||
            state.period !== period
        ) {
            return;
        }

        correlationResult = result;
        renderCorrelation();

    } catch (error) {
        if (token !== correlationToken) {
            return;
        }

        correlationResult = {
            error: error.message
        };

        renderCorrelation();
    }
}

/* Correlation Display */

function renderCorrelation() {
    const content = el('aiContent');

    content.querySelector('#correlationCard')?.remove();

    if (
        !correlationResult ||
        !state.compare ||
        state.chartView !== 'trend' ||
        state.mode !== 'interpret'
    ) {
        return;
    }

    const result = correlationResult;

    const card = document.createElement('div');
    card.id = 'correlationCard';
    card.className = 'insight-card';

    const title = document.createElement('div');
    title.className = 'insight-title';
    title.textContent = 'VARIABLE CORRELATION';

    const body = document.createElement('p');
    body.style.whiteSpace = 'pre-line';

    if (result.error) {
        body.textContent = result.error;
    } else {
        body.textContent =
            `Pearson r: ${pretty(result.pearson_r, 3)}\n` +
            `Spearman ρ: ${pretty(result.spearman_rho, 3)}\n` +
            `Overlapping observations: ${result.observations}\n` +
            `Period: ${result.first_year}–${result.last_year}\n` +
            `P-values are unadjusted.\n` +
            `Correlation does not establish causation.`;
    }

    card.append(title, body);

    const aiCard = [...content.children].find(child => {
        return child.querySelector('.ai-explanation');
    });

    if (aiCard) {
        content.insertBefore(card, aiCard);
    } else {
        content.prepend(card);
    }
}

/* Dashboard Initialization */

async function initializeDashboard() {
    const required = [
        'region',
        'indicator',
        'indicator2',
        'toggleIndicator',
        'period',
        'loadStatus',
        'dashboardGrid',
        'trendChart',
        'aiContent',
        'aiHeading',
        'aiSubheading',
        'askForm',
        'question'
    ];

    const missing = required.filter(id => !el(id));

    if (missing.length) {
        console.error(
            'Missing dashboard elements:',
            missing
        );
        return;
    }

    el('region').addEventListener('change', event => {
        state.region = event.target.value;
        refreshIndicators();
        updateDashboard();
    });

    el('indicator').addEventListener('change', event => {
        state.datasetId = event.target.value;
        refreshSecondIndicator();
        updateDashboard();
    });

    el('indicator2').addEventListener('change', event => {
        state.datasetId2 = event.target.value;
        updateDashboard();
    });

    el('toggleIndicator').addEventListener(
        'click',
        toggleComparison
    );

    document.querySelectorAll('.chart-view').forEach(button => {
        button.addEventListener('click', () => setChartView(button.dataset.view));
    });

    el('period').addEventListener('change', event => {
        state.period = event.target.value;
        updateDashboard();
    });

    document.querySelectorAll('.mode').forEach(button => {
        button.addEventListener('click', () => {
            setMode(button.dataset.mode);
        });
    });

    el('askForm').addEventListener(
        'submit',
        submitQuestion
    );

    try {
        const response = await fetch('/api/datasets');

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        catalog = await response.json();

        if (!catalog.length) {
            throw new Error(
                'No NASA datasets found in data/raw'
            );
        }

        const regions = [
            ...new Set(catalog.map(dataset => dataset.region))
        ];

        el('region').replaceChildren(
            ...regions.map(region => {
                return new Option(
                    regionLabels[region] || region,
                    region
                );
            })
        );

        state.region = regions.includes(state.region)
            ? state.region
            : regions[0];

        el('region').value = state.region;

        refreshIndicators();
        await updateDashboard();

    } catch (error) {
        setError(
            `Cannot connect to FastAPI: ${error.message}`
        );
    }
}


/* Startup */

if (document.readyState === 'loading') {
    document.addEventListener(
        'DOMContentLoaded',
        initializeDashboard
    );
} else {
    initializeDashboard();
}
