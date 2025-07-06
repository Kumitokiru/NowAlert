// analytics.js
document.addEventListener('DOMContentLoaded', () => {
  updateCharts();
});

function updateCharts() {
  // Determine role from path
  const path = window.location.pathname;
  let role = 'all';
  if (path.includes('barangay')) role = 'barangay';
  else if (path.includes('cdrrmo')) role = 'cdrrmo';
  else if (path.includes('pnp')) role = 'pnp';

  // Fetch distribution and full analytics
  fetch(`/api/barangay_analytics_data?role=${role}`)
    .then(res => res.json())
    .then(data => {
      renderPie('distChart', data.distribution, 'Incident Distribution');
      renderLine('trendChart', data.trends.labels, data.trends.total, 'Total Incidents');
      renderBar('weatherChart', data.weather, 'Weather Impact');
      renderBar('roadConditionChart', data.road_conditions, 'Road Condition');
      renderPie('vehicleTypesChart', data.vehicle_types, 'Vehicle Types');
      renderBar('driverAgeChart', data.driver_age, 'Driver Age Groups');
      renderPie('driverGenderChart', data.driver_gender, 'Driver Gender');
      renderBar('injuriesChart', data.injuries_by_time.labels, 'Injuries', data.injuries_by_time.data);
      renderBar('fatalitiesChart', data.fatalities_by_time.labels, 'Fatalities', data.fatalities_by_time.data);
    })
    .catch(err => console.error('Analytics load error:', err));
}

// Helpers
function renderPie(canvasId, objData, title) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (window[canvasId]) window[canvasId].destroy();
  window[canvasId] = new Chart(ctx, {
    type: 'pie',
    data: {
      labels: Object.keys(objData),
      datasets: [{ data: Object.values(objData), backgroundColor: generateColors(objData) }]
    },
    options: { responsive: true, title: { display: true, text: title } }
  });
}

function renderLine(canvasId, labels, dataSet, label) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (window[canvasId]) window[canvasId].destroy();
  window[canvasId] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets:[{ label, data: dataSet, borderColor:'#36A2EB', fill:false }] },
    options: { responsive: true }
  });
}

function renderBar(canvasId, objDataOrLabels, title, overrideData=null) {
  let labels, data;
  if (overrideData) {
    labels = objDataOrLabels;
    data = overrideData;
  } else {
    labels = Object.keys(objDataOrLabels);
    data = Object.values(objDataOrLabels);
  }
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (window[canvasId]) window[canvasId].destroy();
  window[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets:[{ label: title, data, backgroundColor: '#36A2EB' }] },
    options: { responsive: true }
  });
}

function generateColors(obj) {
  // simple generator for up to 10 segments
  const palette = ['#FF6384','#36A2EB','#FFCE56','#4BC0C0','#9966FF','#FF9F40','#8A2BE2','#00CED1','#FF4500','#2E8B57'];
  return Object.keys(obj).map((_, i) => palette[i % palette.length]);
}
