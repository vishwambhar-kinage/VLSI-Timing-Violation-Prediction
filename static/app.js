const pages = ["dashboard","predict","models","analysis","dataset","about"];
function showPage(name){
  pages.forEach(p=>document.getElementById("page-"+p).classList.toggle("active-page",p===name));
  document.querySelectorAll(".nav-link").forEach(b=>b.classList.toggle("active",b.dataset.page===name));
  document.getElementById("crumb-name").textContent=({dashboard:"Dashboard",predict:"Predict",models:"Model Comparison",analysis:"Data Analysis (EDA)",dataset:"Dataset",about:"About"})[name];
  window.scrollTo({top:0,behavior:"smooth"});
}
document.querySelectorAll(".nav-link").forEach(b=>b.addEventListener("click",()=>showPage(b.dataset.page)));
document.querySelectorAll("[data-go]").forEach(b=>b.addEventListener("click",()=>showPage(b.dataset.go)));
async function getJSON(url){const r=await fetch(url);if(!r.ok)throw new Error("Unable to load "+url);return r.json()}
function setDonut(el,clean,total){const pct=total?clean/total*100:0;el.style.background=`conic-gradient(#6741d9 0 ${pct}%, #edb2c6 ${pct}% 100%)`}
async function loadDashboard(){
 try{
  const d=await getJSON("/api/dashboard");
  document.getElementById("stat-samples").textContent=d.samples.toLocaleString();
  document.getElementById("stat-features").textContent=d.features;
  document.getElementById("stat-models").textContent=d.models;
  document.getElementById("clean-count").textContent=d.clean.toLocaleString();
  document.getElementById("violation-count").textContent=d.violations.toLocaleString();
  document.getElementById("clean-percent").textContent=Math.round(d.clean/d.samples*100)+"%";
  setDonut(document.getElementById("donut"),d.clean,d.samples);
  document.getElementById("best-model-name").textContent=d.best_model;
  document.getElementById("best-f1").textContent=d.best_f1;
  document.getElementById("data-source").textContent="Data source: "+d.data_source;
  document.getElementById("eda-samples").textContent=d.samples.toLocaleString();
 }catch(e){console.error(e)}
}
async function loadAnalysis(){
 try{
  const a=await getJSON("/api/analysis"),d=a.class_distribution,total=d.no_violation+d.violation;
  document.getElementById("eda-clean").textContent=d.no_violation.toLocaleString();
  document.getElementById("eda-violation").textContent=d.violation.toLocaleString();
  setDonut(document.getElementById("eda-donut"),d.no_violation,total);
  document.getElementById("feature-bars").innerHTML=a.feature_association.map(x=>`<div class="feature-row"><span>${x.feature.replaceAll("_"," ")}</span><div class="bar-track"><div class="bar-fill" style="width:${x.value*100}%"></div></div><b>${x.value.toFixed(2)}</b></div>`).join("");
 }catch(e){console.error(e)}
}
async function loadModels(){
 try{
  const d=await getJSON("/api/models");
  const best=d.best_model;
  document.getElementById("model-rows").innerHTML=d.models.map(m=>`<tr class="${m.name===best?'best-row':''}"><td>${m.name}${m.name===best?' ★':''}</td><td>${m.accuracy.toFixed(3)}</td><td>${m.precision.toFixed(3)}</td><td>${m.recall.toFixed(3)}</td><td>${m.f1.toFixed(3)}</td></tr>`).join("");
  document.getElementById("model-bars").innerHTML=d.models.map(m=>`<div class="bar-row"><span>${m.name}</span><div class="bar-track"><div class="bar-fill" style="width:${m.f1*100}%"></div></div><b>${m.f1.toFixed(2)}</b></div>`).join("");
  document.getElementById("matrix-title").textContent=best;
  [["m00",0,0],["m01",0,1],["m10",1,0],["m11",1,1]].forEach(([id,r,c])=>document.getElementById(id).textContent=d.confusion_matrix[r][c].toLocaleString());
 }catch(e){console.error(e)}
}
document.getElementById("predict-form").addEventListener("submit",async ev=>{
 ev.preventDefault();
 const form=ev.currentTarget,btn=document.getElementById("predict-btn"),payload=Object.fromEntries(new FormData(form).entries());
 btn.disabled=true;btn.textContent="Predicting…";
 try{
  const r=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const d=await r.json();if(!r.ok)throw new Error(d.error||"Prediction request failed");
  document.getElementById("empty-result").style.display="none";
  document.getElementById("prediction-result").style.display="block";
  const violation=d.prediction===1,banner=document.getElementById("result-banner");
  banner.classList.toggle("violation-result",violation);
  document.getElementById("result-check").textContent=violation?"!":"✓";
  document.getElementById("result-label").textContent=d.label;
  document.getElementById("result-description").textContent=violation?"The model identified this path as a potential timing risk.":"The model classified this path as not violating timing.";
  document.getElementById("confidence-value").textContent=d.confidence+"%";
  document.getElementById("confidence-bar").style.width=d.confidence+"%";
  document.getElementById("predicted-class").textContent=d.prediction+" ("+d.label+")";
  document.getElementById("prediction-model").textContent=d.model;
 }catch(err){alert(err.message)}finally{btn.disabled=false;btn.innerHTML='Run Prediction <span>→</span>'}
});
document.getElementById("predict-form").addEventListener("reset",()=>{
 setTimeout(()=>{document.getElementById("prediction-result").style.display="none";document.getElementById("empty-result").style.display="block"},0)
});
loadDashboard();loadAnalysis();loadModels();
