---
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Chirpy 테마 About 페이지 — _tabs/about.md 에 저장하세요
#  수정: 아래 <script> 안의 DATA = { ... } 블록만 바꾸세요
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
layout: page
title: About
icon: fas fa-user
order: 4
---

<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=JetBrains+Mono:wght@400;600&display=swap');
.pf-wrap{max-width:100%;font-size:15px}
.pf-hero{background:#080E14;border-radius:10px;padding:2.8rem 2.4rem;margin-bottom:2rem;position:relative;overflow:hidden}
.pf-hero::before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(0,200,212,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(0,200,212,.05) 1px,transparent 1px);background-size:48px 48px}
.pf-hero::after{content:'';position:absolute;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(0,200,212,.08) 0%,transparent 70%);top:-150px;right:-100px;pointer-events:none}
.pf-hi{position:relative;z-index:1}
.pf-badge{display:inline-flex;align-items:center;gap:.4rem;font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#00C8D4;letter-spacing:.15em;border:1px solid rgba(0,200,212,.3);padding:.28rem .65rem;border-radius:3px;margin-bottom:1.4rem}
.pf-name{font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,8vw,6.5rem);line-height:.9;letter-spacing:.03em;color:#F0F6FC;margin-bottom:.9rem}
.pf-name em{color:#00C8D4;font-style:normal}
.pf-role{font-family:'JetBrains Mono',monospace;font-size:.88rem;color:#00C8D4;margin-bottom:.7rem}
.pf-desc{max-width:560px;color:#A8B8C8;font-size:.88rem;line-height:1.9}
.pf-tags{display:flex;flex-wrap:wrap;gap:.38rem;margin-top:1.3rem}
.pf-tag{font-family:'JetBrains Mono',monospace;font-size:.63rem;padding:.22rem .55rem;background:rgba(0,200,212,.07);border:1px solid rgba(0,200,212,.2);color:#00C8D4;border-radius:3px}
.pf-stats{display:flex;flex-wrap:wrap;gap:2rem;margin-top:2.2rem;padding-top:1.4rem;border-top:1px solid #1A2C3D}
.pf-snum{font-family:'Bebas Neue',sans-serif;font-size:2rem;color:#00C8D4;line-height:1}
.pf-slbl{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#7A8A9A;letter-spacing:.1em;text-transform:uppercase;margin-top:.18rem}
.pf-sec{margin-bottom:2.4rem}
.pf-lbl{font-family:'JetBrains Mono',monospace;font-size:.63rem;color:#00C8D4;letter-spacing:.2em;text-transform:uppercase;margin-bottom:.3rem;display:flex;align-items:center;gap:.5rem}
.pf-lbl::before{content:'//';opacity:.45}
.pf-ttl{font-family:'Bebas Neue',sans-serif;font-size:clamp(1.8rem,4vw,2.8rem);line-height:.95;color:#F0F6FC;margin-bottom:1.6rem}
.pf-sg{display:grid;grid-template-columns:repeat(3,1fr);gap:1.5px;background:#1A2C3D;border:1px solid #1A2C3D;border-radius:4px;overflow:hidden}
.pf-scat{background:#0F1C28;padding:1.3rem}
.pf-scatt{font-family:'JetBrains Mono',monospace;font-size:.63rem;color:#00C8D4;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.9rem;padding-bottom:.45rem;border-bottom:1px solid #1A2C3D}
.pf-si{display:flex;align-items:center;justify-content:space-between;margin-bottom:.6rem}
.pf-sn{font-size:.76rem;color:#A8B8C8}
.pf-sbw{width:76px;height:3px;background:#1A2C3D;border-radius:2px;overflow:hidden}
.pf-sb{height:100%;border-radius:2px;background:linear-gradient(90deg,#007A85,#00C8D4);transform-origin:left;animation:pfgrow 1.2s cubic-bezier(.16,1,.3,1) both}
@keyframes pfgrow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.pf-rl{display:flex;flex-direction:column}
.pf-ri{display:grid;grid-template-columns:70px 1fr auto;gap:1.4rem;align-items:start;padding:1.3rem 0;border-bottom:1px solid #1A2C3D;transition:padding-left .2s}
.pf-ri:hover{padding-left:.6rem}
.pf-ry{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#00C8D4}
.pf-rt{font-size:.85rem;color:#F0F6FC;line-height:1.6;margin-bottom:.28rem}
.pf-rv{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#7A8A9A}
.pf-rr{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#7A8A9A;white-space:nowrap}
.pf-aw{display:inline-flex;font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#1a1200;background:#F0A500;padding:.14rem .38rem;border-radius:2px;margin-left:.38rem;font-weight:700}
.pf-pg{display:grid;grid-template-columns:repeat(3,1fr);gap:1.5px;background:#1A2C3D;border:1px solid #1A2C3D;border-radius:4px;overflow:hidden}
.pf-pc{background:#0F1C28;padding:1.3rem;position:relative;overflow:hidden;transition:background .2s}
.pf-pc:hover{background:#122030}
.pf-pc::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#007A85,#00C8D4);transform:scaleX(0);transform-origin:left;transition:transform .3s}
.pf-pc:hover::after{transform:scaleX(1)}
.pf-pnum{font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#007A85;margin-bottom:.7rem}
.pf-ptitle{font-size:.85rem;font-weight:700;color:#F0F6FC;margin-bottom:.28rem;line-height:1.4}
.pf-pper{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#7A8A9A;margin-bottom:.7rem}
.pf-pdesc{font-size:.75rem;color:#A8B8C8;line-height:1.75;margin-bottom:.75rem}
.pf-phi{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#00C8D4;padding:.38rem .55rem;background:rgba(0,200,212,.06);border-left:2px solid #00C8D4;margin-bottom:.75rem;line-height:1.55}
.pf-ptags{display:flex;flex-wrap:wrap;gap:.26rem}
.pf-ptag{font-family:'JetBrains Mono',monospace;font-size:.56rem;color:#7A8A9A;padding:.14rem .36rem;border:1px solid #1A2C3D;border-radius:2px}
.pf-el{display:flex;flex-direction:column;gap:1.5px;background:#1A2C3D;border:1px solid #1A2C3D;border-radius:4px;overflow:hidden}
.pf-ei{background:#0F1C28;padding:1.2rem 1.5rem;display:grid;grid-template-columns:120px 1fr;gap:1.4rem}
.pf-edur{font-family:'Bebas Neue',sans-serif;font-size:1.3rem;color:#00C8D4;line-height:1}
.pf-eper{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#7A8A9A;margin-top:.18rem}
.pf-etitle{font-size:.85rem;font-weight:700;color:#F0F6FC}
.pf-eorg{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#7A8A9A;margin:.1rem 0 .55rem}
.pf-esks{display:flex;flex-wrap:wrap;gap:.26rem;margin-bottom:.45rem}
.pf-esk{font-family:'JetBrains Mono',monospace;font-size:.56rem;color:#00C8D4;padding:.1rem .36rem;border:1px solid rgba(0,200,212,.22);border-radius:2px}
.pf-eproj{font-size:.72rem;color:#7A8A9A;font-style:italic}
.pf-edg{display:grid;grid-template-columns:1fr 1fr;gap:2.5rem}
.pf-edlist{display:flex;flex-direction:column;gap:1.1rem}
.pf-editem{display:flex;gap:.75rem;padding-left:.85rem;border-left:2px solid #1A2C3D;position:relative;transition:border-color .2s}
.pf-editem:hover{border-left-color:#00C8D4}
.pf-editem::before{content:'';position:absolute;left:-5px;top:6px;width:7px;height:7px;border-radius:50%;background:#0D1821;border:2px solid #007A85;transition:background .2s,border-color .2s}
.pf-editem:hover::before{background:#00C8D4;border-color:#00C8D4}
.pf-edper{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#7A8A9A;min-width:105px}
.pf-edsch{font-weight:700;color:#F0F6FC;font-size:.83rem}
.pf-eddept{font-size:.7rem;color:#A8B8C8;margin-top:.1rem}
.pf-clist{display:flex;flex-direction:column;gap:.55rem}
.pf-citem{display:flex;align-items:center;justify-content:space-between;padding:.6rem .8rem;background:#0F1C28;border:1px solid #1A2C3D;border-radius:3px;transition:border-color .2s}
.pf-citem:hover{border-color:#00C8D4}
.pf-cname{font-size:.78rem;color:#F0F6FC}
.pf-cdate{font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#7A8A9A}
.pf-cg{display:grid;grid-template-columns:1fr 1fr;gap:2.5rem;align-items:center}
.pf-cdesc{font-size:.88rem;color:#A8B8C8;line-height:1.9}
.pf-clinks{display:flex;flex-direction:column;gap:.55rem}
.pf-clink{display:flex;align-items:center;gap:.85rem;padding:.7rem .95rem;background:#0F1C28;border:1px solid #1A2C3D;text-decoration:none!important;color:#A8B8C8;transition:all .2s;border-radius:3px}
.pf-clink:hover{border-color:#00C8D4;color:#F0F6FC}
.pf-clbl{font-family:'JetBrains Mono',monospace;font-size:.63rem;color:#00C8D4;min-width:52px;letter-spacing:.1em}
.pf-cval{font-size:.78rem}
.pf-hr{border:none;border-top:1px solid #1A2C3D;margin:2.2rem 0}
.pf-sublbl{font-family:'JetBrains Mono',monospace;font-size:.63rem;color:#00C8D4;letter-spacing:.2em;text-transform:uppercase;display:flex;align-items:center;gap:.5rem;margin-bottom:1rem}
.pf-sublbl::before{content:'//';opacity:.45}
.pf-reveal{opacity:0;transform:translateY(22px);transition:opacity .65s ease,transform .65s ease}
.pf-reveal.pf-vis{opacity:1;transform:none}
@media(max-width:768px){
  .pf-sg,.pf-pg{grid-template-columns:1fr}
  .pf-edg,.pf-cg{grid-template-columns:1fr;gap:1.5rem}
  .pf-ei{grid-template-columns:1fr;gap:.3rem}
  .pf-ri{grid-template-columns:60px 1fr}
  .pf-rr{display:none}
  .pf-hero{padding:2rem 1.4rem}
}
</style>


<div class="pf-wrap">

  <div class="pf-hero pf-reveal" id="pfHero"></div>

  <div class="pf-sec pf-reveal">
    <div class="pf-lbl">Technical Skills</div>
    <div class="pf-ttl">CORE<br>SKILLS</div>
    <div class="pf-sg" id="pfSkills"></div>
  </div>

  <hr class="pf-hr">

  <div class="pf-sec pf-reveal">
    <div class="pf-lbl">Academic Output</div>
    <div class="pf-ttl">RESEARCH &amp;<br>AWARDS</div>
    <div class="pf-rl" id="pfResearch"></div>
  </div>

  <hr class="pf-hr">

  <div class="pf-sec pf-reveal">
    <div class="pf-lbl">Portfolio</div>
    <div class="pf-ttl">PROJECTS</div>
    <div class="pf-pg" id="pfProjects"></div>
  </div>

  <hr class="pf-hr">

  <div class="pf-sec pf-reveal">
    <div class="pf-lbl">Training</div>
    <div class="pf-ttl">EXPERIENCE</div>
    <div class="pf-el" id="pfExp"></div>
  </div>

  <hr class="pf-hr">

  <div class="pf-sec pf-reveal">
    <div class="pf-lbl">Background</div>
    <div class="pf-ttl">EDUCATION &amp;<br>CERTIFICATIONS</div>
    <div id="pfEdu"></div>
  </div>

  <hr class="pf-hr">

  <div class="pf-sec pf-reveal">
    <div class="pf-lbl">Get In Touch</div>
    <div class="pf-ttl">CONTACT</div>
    <div id="pfContact"></div>
  </div>

</div>

<script>
const DATA = {
  name_en:   'RYU SEUNGHO',
  name_ko:   '류승호',
  job_title: 'Data Scientist',
  degree:    'M.S. Data Science',
  tagline:   '데이터가 부족하거나 패턴이 불규칙할 때, 어떻게 하면 틀리지 않는 예측을 할 수 있는가. 그 질문에서 시작해 VAE 기반 합성 데이터 생성과 Conformal Inference 기반 불확실성 정량화까지 직접 설계하고 국내·외 학회에서 검증한 실전형 데이터 사이언티스트.',
  contact: {
    phone:  '010-3406-3079',
    email:  'rst3047@gmail.com',
    github: 'github.com/ryunada',
    blog:   'ryunada.github.io',
  },
  hero_tags: ['Time Series','VAE / Deep Learning','NLP / Topic Modeling','ML Scoring','Anomaly Detection','Conformal Inference'],
  stats: [
    { num: '4',  label: 'Academic Presentations' },
    { num: '6+', label: 'Projects Led' },
    { num: '5',  label: 'Certifications Earned' },
    { num: '★1', label: 'Best Paper Award' },
  ],
  skills: [
    { category: 'Machine Learning', items: [
      { name: 'Random Forest / XGBoost / LGB', pct: 85 },
      { name: 'VAE (Variational Autoencoder)',  pct: 80 },
      { name: 'Ensemble / Stacking',            pct: 82 },
      { name: 'Adaptive Conformal Inference',   pct: 75 },
      { name: 'Object Detection (YOLOv4)',      pct: 68 },
    ]},
    { category: 'Programming & Tools', items: [
      { name: 'Python (pandas / sklearn / TF)',    pct: 88 },
      { name: 'SQL (Oracle / PostgreSQL / MySQL)', pct: 78 },
      { name: 'SAS (Certified Specialist)',        pct: 72 },
      { name: 'Git / Docker / Notion',             pct: 74 },
      { name: 'HTML / CSS / JavaScript',           pct: 60 },
    ]},
    { category: 'NLP & Statistics', items: [
      { name: 'DTM / Dynamic Topic Modeling',    pct: 76 },
      { name: 'KoNLPy / Tokenization',           pct: 73 },
      { name: 'Time Series (SL/Fourier/Kalman)', pct: 80 },
      { name: 'Cluster Analysis (k-means/ASW)',  pct: 72 },
      { name: 'Statistical Hypothesis Testing',  pct: 78 },
    ]},
  ],
  research: [
    { year: '2025', title: 'A novel approach for forecasting non-stationary time series: Utilization of a variational autoencoder reflecting seasonal patterns', venue: 'Bayes Comp 2025 · 국제학회', role: '제2저자', award: '' },
    { year: '2024', title: 'VAE-based Replication and Ensemble Methods for Enhanced Time Series Prediction', venue: '한국데이터정보과학회 추계학술발표회', role: '제1저자', award: '우수상' },
    { year: '2024', title: "TED Talk's Topic Variation Utilizing a Dynamic Topic Modeling Approach", venue: '한국통계학회 하계학술논문발표회', role: '제1저자', award: '' },
    { year: '2023', title: 'Prediction Improvement of Non-Stationary Time Series Analysis Based on Transformation', venue: '한국통계학회 동계학술논문발표회', role: '제1저자', award: '' },
  ],
  projects: [
    { num: '01', period: '2024.10-11', role: '석사 연구 / 제1저자', title: 'VAE 기반 시계열 복제 및 앙상블 예측', type: 'Research · 한국데이터정보과학회 2024', desc: '데이터 희소 환경에서 해빙·대기질 시계열의 계절성을 VAE로 합성 재현하고, Stacking 앙상블과 ACI로 예측 불확실성까지 정량화한 연구.', highlight: 'RMSE 0.169→0.141 (16.5% 개선) / 학술대회 우수상', tags: ['VAE', 'Kalman', 'Stacking', 'ACI'] },
    { num: '02', period: '2025.03-06', role: '석사 연구 / 제2저자', title: '계절성 VAE 비정상 시계열 예측 모형', type: 'Research · Bayes Comp 2025', desc: 'Fourier Prior를 VAE 잠재 공간에 통합해 복잡한 계절 패턴 시계열을 정확하게 예측하는 모형 설계.', highlight: 'RMSE 16.5% 개선 / 국제학회 발표', tags: ['Fourier Prior', 'VAE', '1D CNN'] },
    { num: '03', period: '2022.11-12', role: '4~5인 팀장', title: '이미지 인식 기반 레시피 추천 시스템', type: 'Team Project', desc: 'CNN 한계를 YOLOv4-tiny로 극복. 크롤링 데이터셋 구축부터 LightGBM 추천까지 완성.', highlight: 'Accuracy 2%p 향상 / 다중 식재료 실시간 탐지', tags: ['YOLOv4-tiny', 'LightGBM', 'GridSearchCV'] },
    { num: '04', period: '2022.11', role: '4인 팀장', title: '배틀그라운드 순위 예측 모형', type: 'Team Project', desc: '440만 건 게임 로그에서 Feature Engineering과 이상치 제거로 승률 결정 요인 규명.', highlight: '440만 건 처리 / KillPlace 요인 입증', tags: ['Feature Eng.', '440만건', '29변수'] },
    { num: '05', period: '2022.03-04', role: '4~5인 팀장', title: '서울시 취약계층 노인 복지 분석', type: 'Team Project · 공공데이터', desc: 'k-means(k=3) 군집분석으로 서울 자치구를 복지 유형별 분류. 정책 사각지대 식별.', highlight: '복지 사각지대 자치구 군집 식별', tags: ['k-means', 'ASW', '공공데이터'] },
    { num: '06', period: '2022.08', role: '개인', title: '전국 강수량 60년 분석 & 시각화', type: 'Individual Project', desc: '60년치 강수량 공공데이터 시나리오 분석 및 동적 시각화로 기후 트렌드 분석.', highlight: '60년 기후 트렌드 동적 시각화', tags: ['Python', '시나리오분석', '시각화'] },
  ],
  experience: [
    { hours: '960H', period: '2022.06-12', title: '기업 맞춤형 빅데이터 분석가 양성과정', org: '고용노동부 · 한국품질재단', skills: ['Python', 'Oracle SQL', 'PostgreSQL', 'ML(DT·RF·XGB)', 'TensorFlow', '데이터 시각화'], note: '강수량 시각화 · COVID-19 · Iowa 주류 판매 · 뉴스 분류 · 배그 예측 · 레시피 추천' },
    { hours: '200H', period: '2021.04-07', title: 'SW 비전공자를 위한 웹개발 양성과정', org: 'SW융합교육원', skills: ['HTML/CSS/JS', 'Github', '포토샵'], note: '홈쇼핑 사이트 제작 (팀프로젝트 우수상)' },
    { hours: '60H',  period: '2021.09-11', title: '디지털 마케팅 데이터 분석 시각화', org: 'SW융합교육원', skills: ['Python 기초', '텍스트 마이닝', '크롤링'], note: '4차 산업혁명 빅데이터 관심도 변화 분석' },
    { hours: '24H',  period: '2022.05',    title: 'MySQL 데이터베이스 심화과정', org: 'SW융합교육원', skills: ['MySQL', 'WorkBench', 'SELECT·JOIN·서브쿼리'], note: '' },
  ],
  education: [
    { period: '2023.03-2025.02', school: '안동대학교 대학원', dept: '데이터사이언스학과 · 석사' },
    { period: '2017.03-2023.02', school: '안동대학교',        dept: '정보통계학과 (복수: 컴퓨터공학과) · 학사' },
    { period: '2014.03-2017.02', school: '밀양고등학교',      dept: '' },
  ],
  military: { rank: '육군 병장', note: '만기전역 · 조직 책임감 · 역할 분담 체화' },
  certs: [
    { name: '빅데이터분석기사',                date: '2023.07' },
    { name: '정보처리기사',                    date: '2022.03' },
    { name: 'SQL 개발자 (SQLD)',               date: '2022.06' },
    { name: 'SAS Certified Specialist (Base)', date: '2021.07' },
    { name: '사회조사분석사 2급',              date: '2019.05' },
  ],
  contact_desc: 'ML/AI 스코어링, 시계열·이상 탐지, 합성 데이터 생성, 불확실성 정량화 분야에서 함께 일하고 싶으신 분은 언제든지 연락 주세요.',
};

function pesc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function pel(tag,cls,inner,sty){return '<'+tag+(cls?' class="'+cls+'"':'')+(sty?' style="'+sty+'"':'')+'>'+inner+'</'+tag+'>';}
function pfInit(){
  var d=DATA;
  var pts=d.name_en.split(' ');
  var nameH=pts.map(function(w,i){return i===1?'<em>'+pesc(w)+'</em>':pesc(w);}).join('<br>');
  var tags=d.hero_tags.map(function(x){return pel('span','pf-tag',pesc(x));}).join('');
  var stats=d.stats.map(function(s){return pel('div','',pel('div','pf-snum',pesc(s.num))+pel('div','pf-slbl',pesc(s.label)));}).join('');
  document.getElementById('pfHero').innerHTML=
    '<div class="pf-hi">'+
    '<div class="pf-badge">'+pesc(d.job_title)+' &middot; '+pesc(d.name_ko)+'</div>'+
    '<div class="pf-name">'+nameH+'</div>'+
    pel('div','pf-role','// '+pesc(d.job_title)+' &middot; '+pesc(d.degree))+
    pel('p','pf-desc',pesc(d.tagline))+
    pel('div','pf-tags',tags)+
    pel('div','pf-stats',stats)+
    '</div>';

  document.getElementById('pfSkills').innerHTML=d.skills.map(function(cat){
    var items=cat.items.map(function(it,i){
      return pel('div','pf-si',pel('span','pf-sn',pesc(it.name))+'<div class="pf-sbw"><div class="pf-sb" style="width:'+it.pct+'%;animation-delay:'+(i*.07)+'s"></div></div>');
    }).join('');
    return pel('div','pf-scat',pel('div','pf-scatt',pesc(cat.category))+items);
  }).join('');

  document.getElementById('pfResearch').innerHTML=d.research.map(function(r){
    var aw=r.award?'<span class="pf-aw">&#9733; '+pesc(r.award)+'</span>':'';
    return pel('div','pf-ri',
      pel('div','pf-ry',pesc(r.year))+
      pel('div','',pel('div','pf-rt',pesc(r.title)+aw)+pel('div','pf-rv',pesc(r.venue)))+
      pel('div','pf-rr',pesc(r.role)));
  }).join('');

  document.getElementById('pfProjects').innerHTML=d.projects.map(function(p){
    var tg=p.tags.map(function(t){return pel('span','pf-ptag',pesc(t));}).join('');
    var hi=pesc(p.highlight).replace(/ / /g,'<br>');
    return pel('div','pf-pc',
      pel('div','pf-pnum',pesc(p.num)+' &middot; '+pesc(p.period)+' &middot; '+pesc(p.role))+
      pel('div','pf-ptitle',pesc(p.title))+
      pel('div','pf-pper',pesc(p.type))+
      pel('div','pf-pdesc',pesc(p.desc))+
      pel('div','pf-phi',hi)+
      pel('div','pf-ptags',tg));
  }).join('');

  document.getElementById('pfExp').innerHTML=d.experience.map(function(e){
    var sk=e.skills.map(function(s){return pel('span','pf-esk',pesc(s));}).join('');
    return pel('div','pf-ei',
      pel('div','',pel('div','pf-edur',pesc(e.hours))+pel('div','pf-eper',pesc(e.period)))+
      pel('div','',pel('div','pf-etitle',pesc(e.title))+pel('div','pf-eorg',pesc(e.org))+pel('div','pf-esks',sk)+(e.note?pel('div','pf-eproj',pesc(e.note)):'')));
  }).join('');

  var edList=d.education.map(function(e){
    return pel('div','pf-editem',
      pel('div','pf-edper',pesc(e.period))+
      pel('div','',pel('div','pf-edsch',pesc(e.school))+(e.dept?pel('div','pf-eddept',pesc(e.dept)):'')));
  }).join('');
  var mil='<div class="pf-editem" style="border-left-color:#007A85">'+
    '<div class="pf-edper">'+pesc(d.military.rank)+'</div>'+
    '<div><div class="pf-edsch">만기전역</div><div class="pf-eddept">'+pesc(d.military.note)+'</div></div></div>';
  var certList=d.certs.map(function(c){
    return pel('div','pf-citem',pel('span','pf-cname',pesc(c.name))+pel('span','pf-cdate',pesc(c.date)));
  }).join('');
  document.getElementById('pfEdu').innerHTML=
    pel('div','pf-edg',
      pel('div','','<div class="pf-sublbl">Education</div>'+pel('div','pf-edlist',edList)+
        '<div class="pf-sublbl" style="margin-top:1.8rem">Military</div>'+mil)+
      pel('div','','<div class="pf-sublbl">Certifications</div>'+pel('div','pf-clist',certList)));

  var lks=[
    {lbl:'PHONE', val:d.contact.phone,  href:'tel:'+d.contact.phone},
    {lbl:'EMAIL', val:d.contact.email,  href:'mailto:'+d.contact.email},
    {lbl:'GITHUB',val:d.contact.github, href:'https://'+d.contact.github},
    {lbl:'BLOG',  val:d.contact.blog,   href:'https://'+d.contact.blog},
  ].map(function(l){
    return '<a class="pf-clink" href="'+l.href+'" target="_blank">'+
      pel('span','pf-clbl',l.lbl)+pel('span','pf-cval',pesc(l.val))+'</a>';
  }).join('');
  document.getElementById('pfContact').innerHTML=
    pel('div','pf-cg',
      pel('div','pf-cdesc',pesc('데이터로 더 정밀하고 신뢰할 수 있는 의사결정을 만드는 것, 그 목표를 함께 이루어갈 기회를 기다리고 있습니다.')+'<br><br>'+pesc(d.contact_desc))+
      pel('div','pf-clinks',lks));

  var obs=new IntersectionObserver(function(entries){
    entries.forEach(function(e){if(e.isIntersecting)e.target.classList.add('pf-vis');});
  },{threshold:.08});
  document.querySelectorAll('.pf-reveal').forEach(function(el){obs.observe(el);});
}
document.addEventListener('DOMContentLoaded',pfInit);
</script>