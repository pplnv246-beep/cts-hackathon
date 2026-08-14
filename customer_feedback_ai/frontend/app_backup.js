async function loadTrendChart() {

    const canvas = getElement("trendChart");

    if (!canvas) {
        return;
    }

    try {

        const result = await fetchAPI("/analytics/trends");

        const data = result.data || [];

        if (trendChart) {
            trendChart.destroy();
            trendChart = null;
        }

        if (!data.length) {

            const parent = canvas.parentElement;

            if (parent) {

                let message =
                    parent.querySelector(".trend-empty-message");

                if (!message) {

                    message = document.createElement("div");

                    message.className =
                        "trend-empty-message";

                    message.style.padding = "40px";
                    message.style.textAlign = "center";
                    message.style.color = "#64748b";
                    message.style.fontSize = "16px";

                    parent.appendChild(message);
                }

                message.innerHTML =
                    "📊 <strong>Trend analysis unavailable</strong><br>" +
                    "<span style='font-size:14px'>" +
                    (result.message ||
                        "This uploaded CSV does not contain a valid review date column.") +
                    "</span>";

            }

            canvas.style.display = "none";

            return;
        }

        canvas.style.display = "block";

        const oldMessage =
            canvas.parentElement?.querySelector(
                ".trend-empty-message"
            );

        if (oldMessage) {
            oldMessage.remove();
        }

        const labels =
            data.map(
                item => item.month
            );

        const totalReviews =
            data.map(
                item =>
                    Number(
                        item.reviews || 0
                    )
            );

        const negativeReviews =
            data.map(
                item =>
                    Number(
                        item.negative_reviews || 0
                    )
            );

        trendChart =
            new Chart(
                canvas,
                {
                    type: "line",

                    data: {

                        labels: labels,

                        datasets: [

                            {
                                label:
                                    "Total Reviews",

                                data:
                                    totalReviews,

                                tension: 0.3,

                                fill: false
                            },

                            {
                                label:
                                    "Negative Reviews",

                                data:
                                    negativeReviews,

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

    } catch (error) {

        console.error(
            "Trend chart error:",
            error
        );
    }
}
