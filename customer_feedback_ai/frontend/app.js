const API_BASE = window.location.protocol.startsWith("http")
    ? window.location.origin
    : "http://127.0.0.1:8000";

let sentimentChart = null;
let concernChart = null;
let concernSentimentChart = null;
let trendChart = null;


// ============================================================
// PARALLAX SCROLLING EFFECT
// ============================================================

window.addEventListener("scroll", () => {
    const scrolled = window.pageYOffset;
    const orb1 = document.querySelector(".glow-orb-1");
    const orb2 = document.querySelector(".glow-orb-2");
    const orb3 = document.querySelector(".glow-orb-3");
    const rays = document.querySelector(".ambient-light-rays");

    if (orb1) orb1.style.transform = `translate3d(0, ${scrolled * 0.15}px, 0)`;
    if (orb2) orb2.style.transform = `translate3d(0, ${scrolled * -0.12}px, 0)`;
    if (orb3) orb3.style.transform = `translate3d(0, ${scrolled * 0.08}px, 0)`;
    if (rays) rays.style.transform = `translate3d(0, ${scrolled * 0.05}px, 0) rotate(-10deg)`;
});


// ============================================================
// ANIME.JS MOTION ENGINE INITIALIZATION
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    if (typeof anime !== "undefined") {
        anime({
            targets: ".upload-panel, .card, .review-panel, .panel, .insights",
            translateY: [24, 0],
            opacity: [0, 1],
            delay: anime.stagger(100, { start: 150 }),
            duration: 800,
            easing: "easeOutCubic"
        });

        anime({
            targets: ".header-3d-hero",
            scale: [1, 1.05, 1],
            rotate: [0, 2, 0],
            duration: 6000,
            easing: "easeInOutSine",
            loop: true
        });

        document.querySelectorAll("button").forEach((btn) => {
            btn.addEventListener("mouseenter", () => {
                anime({
                    targets: btn,
                    scale: 1.03,
                    duration: 300,
                    easing: "easeOutQuad"
                });
            });
            btn.addEventListener("mouseleave", () => {
                anime({
                    targets: btn,
                    scale: 1.0,
                    duration: 300,
                    easing: "easeOutQuad"
                });
            });
        });
    }
});


// ============================================================
// API HELPER & HEALTH CHECK
// ============================================================

async function checkAPIHealth() {
    const statusElem = document.querySelector(".status");
    if (!statusElem) return;
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            statusElem.innerHTML = `<span class="status-dot"></span> API Connected`;
            statusElem.style.color = "#16a34a";
        } else {
            statusElem.innerHTML = `<span class="status-dot" style="background-color: #ef4444;"></span> API Error`;
            statusElem.style.color = "#dc2626";
        }
    } catch (e) {
        statusElem.innerHTML = `<span class="status-dot" style="background-color: #ef4444;"></span> API Offline`;
        statusElem.style.color = "#dc2626";
    }
}

async function fetchAPI(endpoint) {

    const response = await fetch(
        `${API_BASE}${endpoint}`
    );

    if (!response.ok) {

        const errorText = await response.text();

        throw new Error(
            `${endpoint} returned ${response.status}: ${errorText}`
        );
    }

    return await response.json();
}


// ============================================================
// OVERVIEW
// ============================================================

async function loadOverview() {

    const data = await fetchAPI(
        "/analytics/overview"
    );

    document.getElementById(
        "totalReviews"
    ).textContent =
        Number(
            data.total_reviews
        ).toLocaleString();

    document.getElementById(
        "negativeReviews"
    ).textContent =
        Number(
            data.negative
        ).toLocaleString();

    document.getElementById(
        "positiveReviews"
    ).textContent =
        Number(
            data.positive
        ).toLocaleString();

    document.getElementById(
        "neutralReviews"
    ).textContent =
        Number(
            data.neutral
        ).toLocaleString();

    document.getElementById(
        "negativePercentage"
    ).textContent =
        `${data.negative_percentage}%`;

    document.getElementById(
        "positivePercentage"
    ).textContent =
        `${data.positive_percentage}%`;

    document.getElementById(
        "neutralPercentage"
    ).textContent =
        `${data.neutral_percentage}%`;
}


// ============================================================
// SENTIMENT CHART
// ============================================================

// ============================================================
// SENTIMENT CHART
// ============================================================

// ============================================================
// SENTIMENT CHART (DONUT OVERVIEW WITH PERMANENT SLICE LABELS)
// ============================================================

async function loadSentimentChart() {

    const result = await fetchAPI(
        "/analytics/sentiment"
    );

    const data = result.data || [];

    const labels = data.map(
        item => item.sentiment
    );

    const values = data.map(
        item => item.count
    );

    const colorMap = {
        "Negative": "#EF4444",
        "Positive": "#10B981",
        "Neutral": "#F59E0B"
    };

    const backgroundColors = labels.map(label => colorMap[label] || "#3B82F6");

    if (sentimentChart) {
        sentimentChart.destroy();
    }

    const donutSliceDataLabelsPlugin = {
        id: "donutSliceDataLabels",
        afterDatasetsDraw(chart) {
            const { ctx, chartArea } = chart;
            if (!chartArea) return;
            const dataset = chart.data.datasets[0];
            const total = dataset.data.reduce((a, b) => a + b, 0);

            // Center Text
            const { width, height, top, left } = chartArea;
            ctx.save();
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            const centerX = left + width / 2;
            const centerY = top + height / 2;

            ctx.font = "bold 22px 'Segoe UI', sans-serif";
            ctx.fillStyle = "#1E293B";
            ctx.fillText(total.toLocaleString(), centerX, centerY - 8);

            ctx.font = "600 13px 'Segoe UI', sans-serif";
            ctx.fillStyle = "#64748B";
            ctx.fillText("Total Reviews", centerX, centerY + 14);

            // Permanent slice labels (percentage & count)
            ctx.font = "bold 12px 'Segoe UI', sans-serif";
            ctx.fillStyle = "#FFFFFF";

            const meta = chart.getDatasetMeta(0);
            meta.data.forEach((element, index) => {
                const val = dataset.data[index];
                if (val && total > 0) {
                    const pct = ((val / total) * 100).toFixed(2);
                    if (pct > 5) {
                        const pos = element.tooltipPosition();
                        ctx.fillText(`${pct}% (${val.toLocaleString()})`, pos.x, pos.y);
                    }
                }
            });
            ctx.restore();
        }
    };

    sentimentChart = new Chart(
        document.getElementById(
            "sentimentChart"
        ),
        {
            type: "doughnut",

            data: {
                labels: labels,

                datasets: [
                    {
                        data: values,
                        backgroundColor: backgroundColors,
                        borderColor: "#FFFFFF",
                        borderWidth: 3,
                        hoverOffset: 6
                    }
                ]
            },

            plugins: [donutSliceDataLabelsPlugin],

            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "68%",
                plugins: {
                    legend: {
                        display: true,
                        position: "bottom",
                        labels: {
                            font: { size: 13, weight: "bold", family: "'Segoe UI', sans-serif" },
                            color: "#1E293B",
                            padding: 16,
                            usePointStyle: true,
                            pointStyle: "circle"
                        }
                    },
                    tooltip: {
                        enabled: true,
                        backgroundColor: "#1E293B",
                        titleFont: { size: 13, weight: "bold" },
                        bodyFont: { size: 12 },
                        padding: 10,
                        cornerRadius: 6,
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const val = context.raw || 0;
                                const pct = total > 0 ? ((val / total) * 100).toFixed(2) : 0;
                                return ` ${context.label}: ${val.toLocaleString()} (${pct}%)`;
                            }
                        }
                    }
                }
            }
        }
    );
}


// ============================================================
// CONCERN CHART (HORIZONTAL BAR WITH PERMANENT EDGE LABELS)
// ============================================================

async function loadConcernChart() {

    const result = await fetchAPI(
        "/analytics/concerns"
    );

    const data =
        (result.data || []).slice(0, 8);

    const labels = data.map(
        item => item.concern
    );

    const values = data.map(
        item => item.count
    );

    if (concernChart) {
        concernChart.destroy();
    }

    const barEdgeDataLabelsPlugin = {
        id: "barEdgeDataLabels",
        afterDatasetsDraw(chart) {
            const { ctx } = chart;
            ctx.save();
            ctx.font = "bold 12px 'Segoe UI', sans-serif";
            ctx.fillStyle = "#1E293B";
            ctx.textBaseline = "middle";

            const meta = chart.getDatasetMeta(0);
            meta.data.forEach((bar, index) => {
                const val = chart.data.datasets[0].data[index];
                if (val !== undefined && val !== null) {
                    ctx.fillText(Number(val).toLocaleString(), bar.x + 8, bar.y);
                }
            });
            ctx.restore();
        }
    };

    concernChart = new Chart(
        document.getElementById(
            "concernChart"
        ),
        {
            type: "bar",

            data: {
                labels: labels,

                datasets: [
                    {
                        label: "Reviews",
                        data: values,
                        backgroundColor: "#3B82F6",
                        borderColor: "#2563EB",
                        borderWidth: 1,
                        borderRadius: 6,
                        borderSkipped: false
                    }
                ]
            },

            plugins: [barEdgeDataLabelsPlugin],

            options: {
                indexAxis: "y",

                responsive: true,

                maintainAspectRatio: false,

                layout: {
                    padding: {
                        right: 55
                    }
                },

                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: true,
                        backgroundColor: "#1E293B",
                        titleFont: { size: 13, weight: "bold" },
                        bodyFont: { size: 12 },
                        padding: 10,
                        cornerRadius: 6,
                        callbacks: {
                            label: function(context) {
                                return ` Reviews: ${context.raw.toLocaleString()}`;
                            }
                        }
                    }
                },

                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "Review Frequency",
                            color: "#1E293B",
                            font: { size: 13, weight: "bold", family: "'Segoe UI', sans-serif" },
                            padding: { top: 6, bottom: 0 }
                        },
                        grid: { display: false },
                        ticks: { font: { size: 12, weight: "600" }, color: "#1E293B" },
                        beginAtZero: true
                    },
                    y: {
                        grid: { color: "#F1F5F9", borderDash: [4, 4] },
                        ticks: { font: { size: 13, weight: "500" }, color: "#1E293B" }
                    }
                }
            }
        }
    );
}


// ============================================================
// CONCERN VS SENTIMENT (GROUPED BAR WITH PERMANENT TOP LABELS)
// ============================================================

async function loadConcernSentimentChart() {

    const result = await fetchAPI(
        "/analytics/concern-sentiment"
    );

    const data = result.data || [];

    const labels = data.map(
        item => item.concern
    );

    if (concernSentimentChart) {
        concernSentimentChart.destroy();
    }

    const groupedBarTopDataLabelsPlugin = {
        id: "groupedBarTopDataLabels",
        afterDatasetsDraw(chart) {
            const { ctx } = chart;
            ctx.save();
            ctx.font = "bold 11px 'Segoe UI', sans-serif";
            ctx.fillStyle = "#1E293B";
            ctx.textAlign = "center";
            ctx.textBaseline = "bottom";

            chart.data.datasets.forEach((dataset, datasetIndex) => {
                const meta = chart.getDatasetMeta(datasetIndex);
                meta.data.forEach((bar, index) => {
                    const val = dataset.data[index];
                    if (val && val > 0) {
                        ctx.fillText(Number(val).toLocaleString(), bar.x, bar.y - 3);
                    }
                });
            });
            ctx.restore();
        }
    };

    concernSentimentChart =
        new Chart(
            document.getElementById(
                "concernSentimentChart"
            ),
            {
                type: "bar",

                data: {
                    labels: labels,

                    datasets: [

                        {
                            label: "Negative",

                            data: data.map(
                                item =>
                                    item.negative || 0
                            ),
                            backgroundColor: "#EF4444",
                            borderRadius: 4
                        },

                        {
                            label: "Positive",

                            data: data.map(
                                item =>
                                    item.positive || 0
                            ),
                            backgroundColor: "#10B981",
                            borderRadius: 4
                        },

                        {
                            label: "Neutral",

                            data: data.map(
                                item =>
                                    item.neutral || 0
                            ),
                            backgroundColor: "#F59E0B",
                            borderRadius: 4
                        }

                    ]
                },

                plugins: [groupedBarTopDataLabelsPlugin],

                options: {
                    responsive: true,

                    maintainAspectRatio: false,

                    layout: {
                        padding: {
                            top: 18
                        }
                    },

                    plugins: {
                        legend: {
                            display: true,
                            position: "top",
                            labels: {
                                font: { size: 13, weight: "bold", family: "'Segoe UI', sans-serif" },
                                color: "#1E293B",
                                padding: 16,
                                usePointStyle: true,
                                pointStyle: "circle"
                            }
                        },
                        tooltip: {
                            enabled: true,
                            backgroundColor: "#1E293B",
                            titleFont: { size: 13, weight: "bold" },
                            bodyFont: { size: 12 },
                            padding: 10,
                            cornerRadius: 6
                        }
                    },

                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: "Concern Topics",
                                color: "#1E293B",
                                font: { size: 13, weight: "bold", family: "'Segoe UI', sans-serif" },
                                padding: { top: 6, bottom: 0 }
                            },
                            grid: { display: false },
                            ticks: { font: { size: 12, weight: "600" }, color: "#1E293B" }
                        },
                        y: {
                            title: {
                                display: true,
                                text: "Number of Reviews",
                                color: "#1E293B",
                                font: { size: 13, weight: "bold", family: "'Segoe UI', sans-serif" },
                                padding: { top: 0, bottom: 6 }
                            },
                            grid: { color: "#F1F5F9", borderDash: [4, 4] },
                            ticks: { font: { size: 12, weight: "600" }, color: "#475569" },
                            beginAtZero: true
                        }
                    }
                }
            }
        );
}


// ============================================================
// TREND CHART (CUSTOMER FEEDBACK TRENDS LINE CHART)
// ============================================================

async function loadTrendChart() {

    const result = await fetchAPI(
        "/analytics/trends"
    );

    const data = result.data || [];

    const labels = data.map(
        item => item.month
    );

    const totalReviews = data.map(
        item => item.reviews
    );

    const negativeReviews = data.map(
        item => item.negative_reviews || 0
    );

    if (trendChart) {
        trendChart.destroy();
    }

    trendChart = new Chart(
        document.getElementById(
            "trendChart"
        ),
        {
            type: "line",

            data: {
                labels: labels,

                datasets: [

                    {
                        label: "Total Reviews",

                        data: totalReviews,

                        borderColor: "#2563EB",

                        backgroundColor: "rgba(37, 99, 235, 0.1)",

                        borderWidth: 2,

                        tension: 0.3,

                        fill: true
                    },

                    {
                        label: "Negative Reviews",

                        data: negativeReviews,

                        borderColor: "#EF4444",

                        backgroundColor: "rgba(239, 68, 68, 0.1)",

                        borderWidth: 2,

                        tension: 0.3,

                        fill: true
                    }

                ]
            },

            options: {
                responsive: true,

                maintainAspectRatio: false,

                plugins: {
                    legend: {
                        display: true,
                        position: "top",
                        labels: {
                            font: { size: 13, weight: "bold", family: "'Segoe UI', sans-serif" },
                            color: "#1E293B",
                            padding: 16,
                            usePointStyle: true,
                            pointStyle: "circle"
                        }
                    },
                    tooltip: {
                        enabled: true,
                        backgroundColor: "#1E293B",
                        titleFont: { size: 13, weight: "bold" },
                        bodyFont: { size: 12 },
                        padding: 10,
                        cornerRadius: 6
                    }
                },

                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "Timeline",
                            color: "#1E293B",
                            font: { size: 13, weight: "bold", family: "'Segoe UI', sans-serif" },
                            padding: { top: 6, bottom: 0 }
                        },
                        grid: { display: true, color: "#f1f5f9" },
                        ticks: {
                            font: { size: 12, weight: "600" },
                            color: "#1E293B"
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: "Review Count",
                            color: "#1E293B",
                            font: { size: 13, weight: "bold", family: "'Segoe UI', sans-serif" },
                            padding: { top: 0, bottom: 6 }
                        },
                        grid: { color: "#F1F5F9" },
                        ticks: { font: { size: 12, weight: "600" }, color: "#475569" },
                        beginAtZero: true
                    }
                }
            }
        }
    );
}


// ============================================================
// BUSINESS INSIGHTS
// ============================================================

async function loadInsights() {

    const result = await fetchAPI(
        "/analytics/concerns"
    );

    const concerns = result.data || [];

    const container =
        document.getElementById(
            "insightsContainer"
        );

    if (!container) return;

    container.innerHTML = "";

    if (concerns.length === 0) {

        container.innerHTML =
            `<div class="insight">
                No customer concerns detected.
            </div>`;

        return;
    }

    const ordinalLabels = [
        "Top Customer Concern",
        "Second Major Concern",
        "Third Major Concern",
        "Fourth Major Concern",
        "Fifth Major Concern",
        "Sixth Major Concern",
        "Seventh Major Concern",
        "Eighth Major Concern",
        "Ninth Major Concern",
        "Tenth Major Concern"
    ];

    const topConcern = concerns[0];

    // Card 1: Top Customer Concern
    container.innerHTML +=
        `<div class="insight">
            <strong class="insight-label">Top Customer Concern:</strong>
            <span class="insight-value">${escapeHTML(topConcern.concern)}</span>
            —
            <span class="insight-value">${Number(topConcern.count).toLocaleString()} reviews.</span>
        </div>`;

    // Card 2: Concern Percentage
    container.innerHTML +=
        `<div class="insight">
            <strong class="insight-label">Concern Percentage:</strong>
            <span class="insight-value">${topConcern.percentage}%</span>
            of all reviews mention
            <span class="insight-value">${escapeHTML(topConcern.concern)}.</span>
        </div>`;

    // Cards 3+: Subsequent concerns
    for (let i = 1; i < concerns.length; i++) {
        const item = concerns[i];
        const label = i < ordinalLabels.length ? ordinalLabels[i] : `Major Concern #${i + 1}`;

        container.innerHTML +=
            `<div class="insight">
                <strong class="insight-label">${label}:</strong>
                <span class="insight-value">${escapeHTML(item.concern)}</span>
                —
                <span class="insight-value">${Number(item.count).toLocaleString()} reviews.</span>
            </div>`;
    }
}


// ============================================================
// CSV ANALYSIS
// ============================================================

async function analyzeCSV() {

    const fileInput =
        document.getElementById(
            "csvFile"
        );

    const button =
        document.getElementById(
            "analyzeButton"
        );

    const status =
        document.getElementById(
            "uploadStatus"
        );

    if (
        !fileInput.files ||
        fileInput.files.length === 0
    ) {

        status.textContent =
            "Please select a CSV file.";

        return;
    }

    const file =
        fileInput.files[0];

    const formData =
        new FormData();

    formData.append(
        "file",
        file
    );

    button.disabled = true;

    button.textContent =
        "Analyzing...";

    status.textContent =
        "Uploading and analyzing your dataset. Please wait...";


    try {

        const response =
            await fetch(
                `${API_BASE}/analyze`,
                {
                    method: "POST",
                    body: formData
                }
            );

        const result =
            await response.json();

        if (!response.ok) {

            throw new Error(
                result.detail ||
                "CSV analysis failed."
            );
        }

        status.textContent =
            `Analysis completed successfully. ${
                Number(
                    result.total_reviews
                ).toLocaleString()
            } reviews analyzed.`;

        await loadDashboard();

        if (typeof loadAISummary === "function") {
            await loadAISummary();
        }

    } catch (error) {

        console.error(
            "CSV analysis failed:",
            error
        );

        status.textContent =
            `Analysis failed: ${error.message}`;

    } finally {

        button.disabled = false;

        button.textContent =
            "Analyze CSV";
    }
}


// ============================================================
// LOAD DASHBOARD
// ============================================================

async function loadDashboard() {

    checkAPIHealth();

    const container =
        document.getElementById(
            "insightsContainer"
        );

    try {

        await loadOverview();

        await loadSentimentChart();

        await loadConcernChart();

        await loadConcernSentimentChart();

        await loadTrendChart();

        await loadInsights();

    } catch (error) {

        console.error(
            "Dashboard loading failed:",
            error
        );

        if (container) {

            container.innerHTML =
                `<div class="insight">

                    <strong>
                        Dashboard data loading failed.
                    </strong>

                    <br><br>

                    ${error.message}

                </div>`;
        }
    }
}


// ============================================================
// START
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        console.log(
            "Customer Feedback AI frontend loaded"
        );

        loadDashboard();

    }
);