// main.js: handle fetching APIs and drawing charts
// homepage: load summary and monthly trend
async function loadHome(){
const s = await fetchJSON('/api/summary');
if(s){
document.getElementById('card-pred').innerText = s.predicted !== null ? s.predicted : s.last_cases;
const deltaEl = document.getElementById('delta');
const arrowEl = document.getElementById('arrow');
if(s.trend === 'up') { arrowEl.innerText = '▲'; deltaEl.innerText = '+' + s.delta }
else if(s.trend === 'down') { arrowEl.innerText = '▼'; deltaEl.innerText = s.delta }
else { arrowEl.innerText='–'; deltaEl.innerText='0' }
}
const trend = await fetchJSON('/api/monthly_trend');
if(trend && trend.length){
const labels = trend.map(r=>r.ym);
const data = trend.map(r=>r.cases);
const ctx = document.getElementById('monthlyChart').getContext('2d');
new Chart(ctx, { type:'line', data:{ labels, datasets:[{ label:'cases', data, fill:false }] } });
}
}


// yearly
async function loadYearly(){
const data = await fetchJSON('/api/yearly');
if(!data) return;
const labels = data.map(r=>r['ปี']);
const vals = data.map(r=>r.cases);
const ctx = document.getElementById('yearlyChart').getContext('2d');
new Chart(ctx, { type:'bar', data:{ labels, datasets:[{ label:'cases', data:vals }] } });
}


// phayao
async function loadPhayao(){
const districts = await fetchJSON('/api/phayao?level=district');
if(!districts) return;
const list = document.getElementById('district-list');
list.innerHTML = '';
const labels = districts.map(d=>d['อำเภอ']);
const vals = districts.map(d=>d.cases);
// draw district chart
const dctx = document.getElementById('districtChart').getContext('2d');
new Chart(dctx, { type:'bar', data:{ labels, datasets:[{ label:'cases', data:vals }] } });
districts.forEach(d=>{
const li = document.createElement('li'); li.innerText = d['อำเภอ'] + ' — ' + d.cases;
li.onclick = async ()=>{
document.getElementById('sub-title').innerText = 'ตำบลของ ' + d['อำเภอ'];
const subs = await fetchJSON('/api/phayao?level=subdistrict&district=' + encodeURIComponent(d['อำเภอ']));
if(subs && subs.length){
const labelsS = subs.map(s=>s['ตำบล']);
const valsS = subs.map(s=>s.cases);
const sctx = document.getElementById('subChart').getContext('2d');
new Chart(sctx, { type:'bar', data:{ labels:labelsS, datasets:[{ label:'cases', data:valsS }] } });
}
};
list.appendChild(li);
});
}


// init depending on page
window.addEventListener('load', ()=>{
if(document.getElementById('monthlyChart')) loadHome();
if(document.getElementById('yearlyChart')) loadYearly();
if(document.getElementById('districtChart')) loadPhayao();
});