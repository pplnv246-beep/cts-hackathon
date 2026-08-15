const API_BASE = window.location.origin;

let sentimentChart = null;
let concernChart = null;
let concernSentimentChart = null;
let trendChart = null;


// ============================================================
// API HELPER
// ============================================================

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

    if (sentimentChart) {
        sentimentChart.destroy();
    }

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
                        data: values
                    }
                ]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        }
    );
}


// ============================================================
// CONCERN CHART
// ============================================================

async function loadConcernChart() {

    const result = await fetchAPI(
        "/analytics/concerns"
    );

    const data =
        (result.data || []).slice(0, 9);

    const labels = data.map(
        item => item.concern
    );

    const values = data.map(
        item => item.count
    );

    if (concernChart) {
        concernChart.destroy();
    }

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
                        data: values
                    }
                ]
            },

            options: {
                indexAxis: "y",

                responsive: true,

                maintainAspectRatio: false,

                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        }
    );
}


// ============================================================
// CONCERN VS SENTIMENT
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
                            )
                        },

                        {
                            label: "Positive",

                            data: data.map(
                                item =>
                                    item.positive || 0
                            )
                        },

                        {
                            label: "Neutral",

                            data: data.map(
                                item =>
                                    item.neutral || 0
                            )
                        }

                    ]
                },

                options: {
                    responsive: true,

                    maintainAspectRatio: false,

                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            }
        );
}


// ============================================================
// TREND CHART
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

                        tension: 0.3,

                        fill: false
                    },

                    {
                        label: "Negative Reviews",

                        data: negativeReviews,

                        tension: 0.3,

                        fill: false
                    }

                ]
            },

            options: {
                responsive: true,

                maintainAspectRatio: false,

                scales: {
                    y: {
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

    container.innerHTML = "";

    if (concerns.length === 0) {

        container.innerHTML =
            `<div class="insight">
                No customer concerns detected.
            </div>`;

        return;
    }

    const topConcern = concerns[0];

    container.innerHTML +=
        `<div class="insight">

            <strong>
                Top Customer Concern:
            </strong>

            ${topConcern.concern}

            —

            ${Number(
                topConcern.count
            ).toLocaleString()}

            reviews.

        </div>`;


    container.innerHTML +=
        `<div class="insight">

            <strong>
                Concern Percentage:
            </strong>

            ${topConcern.percentage}%

            of all reviews mention

            ${topConcern.concern}.

        </div>`;


    if (concerns.length >= 2) {

        container.innerHTML +=
            `<div class="insight">

                <strong>
                    Second Major Concern:
                </strong>

                ${concerns[1].concern}

                —

                ${Number(
                    concerns[1].count
                ).toLocaleString()}

                reviews.

            </div>`;
    }


    if (concerns.length >= 3) {

        container.innerHTML +=
            `<div class="insight">

                <strong>
                    Third Major Concern:
                </strong>

                ${concerns[2].concern}

                —

                ${Number(
                    concerns[2].count
                ).toLocaleString()}

                reviews.

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
// SINGLE REVIEW ANALYSIS
// ============================================================

async function analyzeReview() {

    console.log(
        "Analyze Review button clicked"
    );

    const reviewInput =
        document.getElementById(
            "reviewInput"
        );

    const button =
        document.getElementById(
            "predictButton"
        );

    const status =
        document.getElementById(
            "predictionStatus"
        );

    const result =
        document.getElementById(
            "predictionResult"
        );

    const review =
        reviewInput.value.trim();

    if (!review) {

        status.textContent =
            "Please enter a customer review.";

        result.style.display =
            "none";

        return;
    }

    button.disabled = true;

    button.textContent =
        "Analyzing...";

    status.textContent =
        "AI is analyzing the review...";

    result.style.display =
        "none";

    try {

        console.log(
            "Sending review to /predict..."
        );

        const response =
            await fetch(
                `${API_BASE}/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        review: review
                    })
                }
            );

        const data =
            await response.json();

        console.log(
            "Prediction response:",
            data
        );

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Prediction failed."
            );
        }

        const sentiment =
            data.sentiment || "Unknown";

        document.getElementById(
            "resultSentiment"
        ).textContent =
            sentiment;

        const confidence =
            Number(
                data.confidence || 0
            );

        document.getElementById(
            "resultConfidence"
        ).textContent =
            `${(
                confidence * 100
            ).toFixed(2)}%`;

        const concerns =
            data.concerns || [];

        document.getElementById(
            "resultConcerns"
        ).textContent =
            concerns.length > 0
                ? concerns.join(", ")
                : "No specific concerns detected";

        const probabilityContainer =
            document.getElementById(
                "probabilityContainer"
            );

        probabilityContainer.innerHTML =
            "";

        const probabilities =
            data.probabilities || {};

        const probabilityOrder = [
            "Negative",
            "Neutral",
            "Positive"
        ];

        probabilityOrder.forEach(
            sentimentName => {

                const value =
                    Number(
                        probabilities[
                            sentimentName
                        ] || 0
                    );

                const percentage =
                    value * 100;

                probabilityContainer.innerHTML +=
                    `
                    <div class="probability-row">

                        <div class="probability-label">

                            <span>
                                ${sentimentName}
                            </span>

                            <span>
                                ${percentage.toFixed(2)}%
                            </span>

                        </div>

                        <div class="probability-bar">

                            <div
                                class="probability-fill"
                                style="width: ${percentage}%"
                            ></div>

                        </div>

                    </div>
                    `;
            }
        );

        result.style.display =
            "block";

        status.textContent =
            "Analysis completed successfully.";

    } catch (error) {

        console.error(
            "Review analysis failed:",
            error
        );

        status.textContent =
            `Analysis failed: ${error.message}`;

        result.style.display =
            "none";

    } finally {

        button.disabled =
            false;

        button.textContent =
            "Analyze Review";
    }
}


// ============================================================
// LOAD DASHBOARD
// ============================================================

async function loadDashboard() {

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