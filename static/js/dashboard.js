// Use the current host for flexibility (local or deployed)
const socket = io(window.location.origin); // e.g., http://localhost:5000 or https://alertnow.onrender.com
const alertContainer = document.getElementById('alert-container');

socket.on('new_alert', (data) => {
    const div = document.createElement('div');
    const now = new Date();
    const uploadTime = data.imageUploadTime ? new Date(data.imageUploadTime) : null;
    const isImageValid = uploadTime && (now - uploadTime) <= 30 * 60 * 1000;

    div.innerHTML = `
        <p><strong>Type:</strong> ${data.emergency_type || 'Not Specified'}</p>
        <p><strong>Location:</strong> ${data.lat}, ${data.lon}</p>
        ${data.image && isImageValid ? `<img src="data:image/jpeg;base64,${data.image}" width="200"/>` : ''}
        <hr>
    `;
    alertContainer.prepend(div);
    updateCharts();
});

function updateCharts() {
    const role = window.location.pathname.includes('barangay') ? 'barangay' : 
                 window.location.pathname.includes('cdrrmo') ? 'cdrrmo' : 
                 window.location.pathname.includes('pnp') ? 'pnp' : 
                 window.location.pathname.includes('bfp') ? 'bfp' : 'all';
    fetch(`/api/distribution?role=${role}`)
        .then(res => res.json())
        .then(dist => {
            const ctxDist = document.getElementById('distChart')?.getContext('2d');
            if (ctxDist && window.distChart) window.distChart.destroy();
            if (ctxDist) {
                window.distChart = new Chart(ctxDist, {
                    type: 'pie',
                    data: {
                        labels: Object.keys(dist),
                        datasets: [{
                            data: Object.values(dist),
                            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }
        });
    const ctxTrend = document.getElementById('trendChart')?.getContext('2d');
    if (ctxTrend && window.trendChart) window.trendChart.destroy();
    if (ctxTrend) {
        window.trendChart = new Chart(ctxTrend, {
            type: 'line',
            data: {
                labels: ['Jun 10', 'Jun 11', 'Jun 12', 'Jun 13', 'Jun 14', 'Jun 15', 'Jun 16'],
                datasets: [{
                    label: 'Incidents',
                    data: [5, 8, 3, 7, 4, 6, 2],
                    borderColor: '#36A2EB',
                    fill: false
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }
}

updateCharts();
