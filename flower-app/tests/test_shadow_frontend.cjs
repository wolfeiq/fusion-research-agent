'use strict';

// UI contract tests with an in-memory DOM and HTTP boundary. No server or RL runs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.listeners = {}; this.dataset = {};
    this.value = ''; this.textContent = ''; this.hidden = false; this.disabled = false;
    this.scrollHeight = 100; this.scrollTop = 0; this.clientHeight = 100;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = [...children]; }
  get childNodes() { return this.children; }
  set innerHTML(_value) { throw new Error('Untrusted HTML must never be assigned'); }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  querySelectorAll(tag) { return this.children.flatMap((child) => [...(child.tag === tag ? [child] : []), ...child.querySelectorAll(tag)]); }
  reportValidity() { return true; }
  focus() {}
  showModal() { this.open = true; }
  close() { this.open = false; }
}

const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'frontend/index.html'), 'utf8');
const script = fs.readFileSync(path.join(root, 'frontend/research.js'), 'utf8');
const flush = () => new Promise(setImmediate);

async function harness({available = true, initialRun, initialResearchRun, researchConfigured = false, researchCatalog} = {}) {
  const elements = {};
  for (const match of html.matchAll(/<([a-z]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    assert(!elements[match[3]], `Duplicate id: ${match[3]}`);
    const element = new Element(match[1]);
    element.value = /\bvalue="([^"]*)"/.exec(match[2])?.[1] || '';
    elements[match[3]] = element;
  }
  const examples = [...html.matchAll(/data-research-example="([^"]+)"/g)].map((match) => {
    const button = new Element('button'); button.dataset.researchExample = match[1]; return button;
  });
  elements['research-shadow-rounds'].value = '1';
  const calls = [], timers = [];
  let generation = 0;
  let shadow = {
    active: false,
    runtime: {available, reason: available ? 'CPU simulator ready' : 'TORAX not installed'},
    catalog: {changes: [
      {id: 'td3_actor_lr', label: 'TD3 actor learning rate', type: 'number', min: 0.000001, max: 0.01, baseline: 0.0001, default: 0.00005},
      {id: 'heating_critic_coordinates', label: 'Heating critic coordinates', type: 'string', options: ['physical', 'normalized'], baseline: 'normalized', default: 'physical'},
    ], limits: {max_seeds: 3, max_rounds: 2, max_seconds: 600}},
    run: initialRun || {id: null, status: 'idle', events: [], comparison: null},
  };
  let research = {active: false, configured: {nebius: researchConfigured, tavily: researchConfigured}, catalog: researchCatalog || [{id: 'saved-study', title: 'Study', kind: 'experiment', status: 'complete'}], run: initialResearchRun || {}};
  const context = {
    URL, URLSearchParams, Set, JSON,
    setTimeout: (fn, delay) => { const timer = {fn, delay, active: true}; timers.push(timer); return timer; },
    clearTimeout: (timer) => { if (timer) timer.active = false; },
    document: {getElementById: (id) => elements[id], createElement: (tag) => new Element(tag), querySelectorAll: () => examples},
    fetch: async (url, options = {}) => {
      calls.push({url, options});
      if (url === '/api/research') return {ok: true, json: async () => research};
      if (url === '/api/shadow/prepare') {
        const config = JSON.parse(options.body);
        shadow = {...shadow, active: false, run: {
          id: `trial${++generation}`, status: 'frozen', config,
          manifest: {baseline: {critic_coordinates: 'normalized', td3_actor_lr: 0.0001}, candidate: config, source_sha256: 'abc123'},
          budget: {episodes: 24, max_seconds: 600}, events: [{kind: 'frozen', message: 'Source captured'}], comparison: null,
        }};
      } else if (url === '/api/shadow/start') {
        assert.equal(JSON.parse(options.body).id, shadow.run.id);
        assert.equal(shadow.run.status, 'frozen');
        shadow = {...shadow, active: true, run: {...shadow.run, status: 'running', events: [{kind: 'start', message: 'Baseline started'}]}};
      } else assert.equal(url, '/api/shadow');
      return {ok: true, json: async () => shadow};
    },
  };
  vm.runInNewContext(script, context, {filename: 'research.js'});
  await flush();
  const get = (name) => elements[`research-shadow-${name}`];
  const fire = async (name, event = 'click') => { get(name).listeners[event]({preventDefault() {}}); await flush(); };
  return {get, fire, calls, timers, elements, examples, setState: (value) => { shadow = value; }, state: () => shadow,
    setResearchState: (value) => { research = {...research, ...value}; }};
}

test('saved API run request is visible before a report and refresh preserves pending form edits', async () => {
  const h = await harness({initialResearchRun: {id: 'api-started', mode: 'live', status: 'running',
    question: 'Recorded question <script>literal text</script>', allow_shadow: true, events: []}});
  assert.equal(h.elements['research-run-context'].hidden, false);
  assert.equal(h.elements['research-saved-question'].textContent, 'Recorded question <script>literal text</script>');
  assert.match(h.elements['research-saved-shadow-permission'].textContent, /enabled for this run/);
  assert.match(h.elements['research-saved-shadow-permission'].textContent, /does not mean a pilot ran/);
  assert.equal(h.elements['research-results'].hidden, true);
  h.elements['research-question'].value = 'My next draft question';
  h.elements['research-allow-shadow'].checked = true;
  h.setResearchState({run: {id: 'other-saved-run', mode: 'live', status: 'failed',
    question: 'A different saved request', allow_shadow: false, events: []}});
  h.elements['research-refresh'].listeners.click(); await flush();
  assert.match(h.elements['research-run-meta'].textContent, /other-saved-run/);
  assert.equal(h.elements['research-saved-question'].textContent, 'A different saved request');
  assert.match(h.elements['research-saved-shadow-permission'].textContent, /disabled for this run/);
  assert.equal(h.elements['research-question'].value, 'My next draft question');
  assert.equal(h.elements['research-allow-shadow'].checked, true);
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
});

test('saved permission distinguishes missing metadata from explicit refusal and offline inspection', async () => {
  const h = await harness({initialResearchRun: {id: 'legacy', mode: 'live', status: 'completed', events: []}});
  assert.equal(h.elements['research-saved-question'].textContent, 'Question not recorded for this saved run.');
  assert.match(h.elements['research-saved-shadow-permission'].textContent, /not recorded/);
  h.setResearchState({run: {id: 'offline', mode: 'offline', allow_shadow: false, events: []}});
  h.elements['research-refresh'].listeners.click(); await flush();
  assert.match(h.elements['research-saved-shadow-permission'].textContent, /unavailable in local inspection/);
  h.setResearchState({run: {}});
  h.elements['research-refresh'].listeners.click(); await flush();
  assert.equal(h.elements['research-run-context'].hidden, true);
  assert.equal(h.elements['research-saved-question'].textContent, '');
  assert.equal(h.elements['research-saved-shadow-permission'].textContent, '');
});

test('fresh clone selects both published pilots without assuming a federation dossier', async () => {
  const pilots = ['993df979e388405297f0195bfca504d6', '43d55b69d02440b3a019cf9b64f6845a'].map((id) => ({
    id: `examples/evidence/shadow/${id}/comparison.json`, kind: 'experiment', status: 'complete', adapter: 'shadow_comparison',
  }));
  const h = await harness({researchCatalog: [
    {id: 'docs/coldstart_coupled_federation.md', kind: 'protocol', status: 'protocol'},
    {id: 'federation:local-smoke', kind: 'experiment', status: 'complete'},
    ...pilots,
  ]});
  const controls = h.elements['research-catalog'].querySelectorAll('input');
  assert.deepEqual(controls.filter((input) => input.checked).map((input) => input.value), pilots.map((row) => row.id));
  assert.equal(controls.find((input) => input.value === 'federation:local-smoke').disabled, false);
  assert.equal(h.elements['research-selection-count'].textContent, '2/8 selected');
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
});

test('original federation dossier defaults remain available without published pilots', async () => {
  const ids = ['federation:local-smoke', 'federation:kaggle-359552340', 'results/td3_heating_stability/repair_comparison.json'];
  const h = await harness({researchCatalog: ids.map((id) => ({id, kind: 'experiment', status: 'complete'}))});
  assert.deepEqual(h.elements['research-catalog'].querySelectorAll('input').filter((input) => input.checked).map((input) => input.value), ids);
});

test('question presets operate on selected experiments and make dossier inspection conditional', async () => {
  const h = await harness();
  const initial = /<textarea id="research-question"[^>]*>([^<]+)<\/textarea>/.exec(html)[1];
  for (const button of h.examples) {
    button.listeners.click();
    const question = h.elements['research-question'].value;
    assert.match(question, /selected experiment evidence/);
    assert.match(question, /[Cc]ite exact/);
    assert.match(question, /only if.*selected/);
    if (button.dataset.researchExample === 'coldstart') assert.equal(question, initial);
  }
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
});

test('startup only reads state; shadow does not require model credentials', async () => {
  const h = await harness();
  assert.deepEqual(h.calls.map((call) => call.url).sort(), ['/api/research', '/api/shadow']);
  assert(h.calls.every((call) => call.options.method === undefined));
  assert.equal(h.get('prepare').disabled, false);
  assert.equal(h.get('start').disabled, true);
  assert.equal(h.elements['research-live'].disabled, true);
  assert.equal(h.get('value').value, '0.00005');
  assert.equal(h.get('seeds').value, '17');
  assert.equal(h.timers.filter((timer) => timer.active).length, 0);
  await h.fire('start');
  assert.equal(h.calls.length, 2);
});

test('freeze and explicit launch send separate, bounded requests', async () => {
  const h = await harness();
  await h.fire('form', 'submit');
  const prepare = h.calls.find((call) => call.url.endsWith('/prepare'));
  assert.deepEqual(JSON.parse(prepare.options.body), {change: 'td3_actor_lr', value: 0.00005, seeds: [17], rounds: 1});
  assert.equal(prepare.options.headers['X-Fusion-UI'], '1');
  assert.equal(h.state().run.status, 'frozen');
  assert.equal(h.get('start').disabled, false);
  assert.equal(h.calls.filter((call) => call.url.endsWith('/start')).length, 0);
  assert.match(h.get('manifest').textContent, /abc123/);
  assert.match(h.get('export').href, /id=trial1/);
  await h.fire('start');
  const start = h.calls.find((call) => call.url.endsWith('/start'));
  assert.deepEqual(JSON.parse(start.options.body), {id: 'trial1'});
  assert.equal(h.get('prepare').disabled, true);
  assert.equal(h.get('start').disabled, true);
  assert.equal(h.get('seeds').disabled, true);
  assert(h.timers.some((timer) => timer.active && timer.delay === 2500));
  assert(h.calls.every((call) => !call.url.startsWith('/api/run')));
});

test('editing a frozen candidate requires refreezing and never launches the old trial', async () => {
  const h = await harness();
  await h.fire('form', 'submit');
  h.get('value').value = '0.000025'; await h.fire('value', 'input');
  assert.equal(h.get('start').disabled, true);
  assert.match(h.get('state-note').textContent, /Settings changed/);
  await h.fire('start');
  assert.equal(h.calls.filter((call) => call.url.endsWith('/start')).length, 0);
  await h.fire('form', 'submit');
  assert.equal(h.get('start').disabled, false);
  await h.fire('start');
  assert.deepEqual(JSON.parse(h.calls.find((call) => call.url.endsWith('/start')).options.body), {id: 'trial2'});
});

for (const seeds of ['', '17,17', '1,2,3,4', '-1', '1.5', '1e3', '1000001', '9007199254740993']) {
  test(`invalid paired seeds rejected locally: ${JSON.stringify(seeds)}`, async () => {
    const h = await harness(); h.get('seeds').value = seeds;
    await h.fire('form', 'submit');
    assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
    assert.equal(h.get('error').hidden, false);
  });
}

for (const value of ['', 'NaN', 'Infinity', '-0.1', '0.1', '0.0001']) {
  test(`invalid or unchanged candidate rejected locally: ${JSON.stringify(value)}`, async () => {
    const h = await harness(); h.get('value').value = value;
    await h.fire('form', 'submit');
    assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
    assert.equal(h.get('error').hidden, false);
  });
}

test('unavailable runtime disables preparation without requiring account setup', async () => {
  const h = await harness({available: false});
  assert.equal(h.get('prepare').disabled, true);
  assert.match(h.get('runtime').textContent, /TORAX not installed/);
  await h.fire('form', 'submit');
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
});

test('enum candidates preserve string values and support paired seed lists', async () => {
  const h = await harness();
  h.get('change').value = 'heating_critic_coordinates'; await h.fire('change', 'change');
  assert.equal(h.get('value').hidden, true);
  assert.equal(h.get('value-option').value, 'physical');
  h.get('seeds').value = '17, 29 43'; h.get('rounds').value = '2';
  await h.fire('form', 'submit');
  const body = JSON.parse(h.calls.find((call) => call.url.endsWith('/prepare')).options.body);
  assert.deepEqual(body, {change: 'heating_critic_coordinates', value: 'physical', seeds: [17, 29, 43], rounds: 2});
});

test('reload resumes a frozen record without launching and completed results remain literal text', async () => {
  const frozen = {id: 'existing', status: 'frozen', config: {change: 'td3_actor_lr', value: 0.000025, seeds: [29], rounds: 2}, manifest: {source_sha256: 'saved'}, budget: {episodes: 48}, events: []};
  const h = await harness({initialRun: frozen});
  assert.equal(h.get('value').value, '0.000025');
  assert.equal(h.get('seeds').value, '29');
  assert.equal(h.get('rounds').value, '2');
  assert.equal(h.get('start').disabled, false);
  h.setState({...h.state(), active: false, run: {...frozen, status: 'completed', comparison: {baseline: {cost: 0.2}, candidate: {cost: 0.3}, recommendation: '<script>Do not execute</script>', failures: []}}});
  await h.fire('refresh');
  assert.equal(h.get('results').hidden, false);
  assert.equal(h.get('recommendation').textContent, 'Pilot recommendation: <script>Do not execute</script>');
  assert.equal(h.get('start').disabled, true);
  assert.equal(h.get('stage-completed').className, 'observed');
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
});

test('failed and timed-out trials cannot be silently relaunched', async () => {
  const h = await harness(); await h.fire('form', 'submit');
  for (const status of ['failed', 'timed_out', 'interrupted']) {
    h.setState({...h.state(), active: false, run: {...h.state().run, status, error: 'Recorded failure'}});
    await h.fire('refresh'); await h.fire('start');
    assert.equal(h.get('start').disabled, true);
    assert.equal(h.get('error').textContent, 'Recorded failure');
  }
  assert.equal(h.calls.filter((call) => call.url.endsWith('/start')).length, 0);
});

test('agent shadow permission is default off, explicit for live, and always false offline', async () => {
  assert.match(html, /id="research-allow-shadow" type="checkbox">/);
  const h = await harness({researchConfigured: true});
  h.elements['research-question'].value = 'Inspect the saved federation evidence';
  const submitResearch = async () => {
    h.elements['research-form'].listeners.submit({preventDefault() {}});
    await flush();
  };
  await submitResearch();
  let posts = h.calls.filter((call) => call.url === '/api/research' && call.options.method === 'POST');
  assert.equal(JSON.parse(posts[0].options.body).allow_shadow, false);
  h.elements['research-allow-shadow'].checked = true;
  await submitResearch();
  posts = h.calls.filter((call) => call.url === '/api/research' && call.options.method === 'POST');
  assert.equal(JSON.parse(posts[1].options.body).allow_shadow, true);
  h.elements['research-offline'].listeners.click(); await flush();
  posts = h.calls.filter((call) => call.url === '/api/research' && call.options.method === 'POST');
  assert.equal(JSON.parse(posts[2].options.body).mode, 'offline');
  assert.equal(JSON.parse(posts[2].options.body).allow_shadow, false);
  assert.equal(h.calls.filter((call) => call.url.endsWith('/start') || call.url.endsWith('/prepare')).length, 0);
});

test('planned episode costs distinguish training from evaluation and are not completed counts', async () => {
  const h = await harness();
  assert.match(h.get('budget-hint').textContent, /56 TORAX episodes \(32 training \+ 24 evaluation\)/);
  h.get('seeds').value = '17,1000000'; h.get('rounds').value = '2'; await h.fire('rounds', 'input');
  assert.match(h.get('budget-hint').textContent, /184 TORAX episodes \(96 training \+ 88 evaluation\)/);
  assert.match(h.get('budget-hint').textContent, /planned, not completed/);
});

function comparisonFixture() {
  const arms = ['local', 'reset_only', 'uniform', 'clipped'];
  const summary = (cost) => ({mean_cost: cost, mean_Te_profile_rmse_keV: 0.03,
    mean_beta_mae: 0.002, valid_complete: true, competent: false});
  const branch = (cost) => ({status: 'complete', accounting: {training_episodes: 16, evaluation_episodes: 12},
    seeds: {'17': {seed: 17, status: 'complete', training_stable: true,
      arms: Object.fromEntries(arms.map((arm) => [arm, {raw: summary(cost), deployed: summary(cost),
        gate: ['uniform', 'clipped'].includes(arm) ? {accept: true, reason: 'pass'} : null}]))}}});
  return {status: 'complete', execution_status: 'completed', expected_training_seeds: [17],
    branches: {baseline: branch(0.1), candidate: branch(0.11)},
    candidate_vs_baseline_raw_endpoint: Object.fromEntries(arms.map((arm) => [arm, {by_seed: [{seed: 17, comparable: true, relative_cost_change: 0.1}]}])),
    recommendation: {text: 'Keep the baseline; this candidate had higher endpoint cost.', code: 'keep_baseline', automatic_apply: false},
    limitations: ['One seed is development evidence only.', 'Endpoint cost is not shots-to-competence.'], failures: [],
    provenance: {trial_id: 'saved-trial', result_sha256: 'test-fingerprint'}};
}

const descendantText = (element) => [element.textContent, ...element.children.map(descendantText)].join(' ');

test('paired raw results render human-readable tables and full JSON is collapsed', async () => {
  const comparison = comparisonFixture();
  const h = await harness({initialRun: {id: 'saved-trial', status: 'completed', comparison, events: []}});
  const area = h.get('comparison');
  const tables = area.querySelectorAll('table');
  assert.equal(tables.length, 2);
  const raw = tables[0];
  assert.match(descendantText(raw), /Baseline → candidate, before any gate fallback/);
  assert.equal(raw.querySelectorAll('tbody')[0].children.length, 4);
  assert.match(descendantText(raw), /0\.10000 0\.11000 \+10\.00%/);
  assert.match(descendantText(raw), /0\.03000 → 0\.03000/);
  assert.match(descendantText(raw), /0\.002000 → 0\.002000/);
  assert.match(descendantText(raw), /competence not reached/);
  assert.equal(h.get('recommendation').textContent, 'Pilot recommendation: ' + comparison.recommendation.text);
  assert.doesNotMatch(h.get('recommendation').textContent, /automatic_apply|keep_baseline/);
  assert.match(descendantText(area), /One seed is development evidence only/);
  const full = area.querySelectorAll('details').find((item) => item.className.includes('full-record'));
  assert(full && !full.open);
  assert.equal(full.querySelectorAll('pre')[0].textContent, JSON.stringify(comparison, null, 2));
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
});

test('gate fallback values remain distinct from raw outcome values', async () => {
  const comparison = comparisonFixture();
  const row = comparison.branches.candidate.seeds['17'].arms.uniform;
  row.gate = {accept: false, reason: 'cost_regression'};
  row.raw.mean_cost = 0.14;
  row.deployed.mean_cost = 0.09;
  const h = await harness({initialRun: {id: 'gated', status: 'completed', comparison, events: []}});
  const [raw, gates] = h.get('comparison').querySelectorAll('table');
  assert.match(descendantText(raw), /0\.14000/);
  assert.doesNotMatch(descendantText(raw), /0\.09000/);
  const candidateGate = gates.querySelectorAll('tbody')[0].children.find((item) => descendantText(item).includes('Rejected'));
  assert.match(descendantText(candidateGate), /Rejected · incumbent fallback \(cost_regression\) 0\.14000 0\.09000/);
});

test('unfinished trials keep missing arms and failures visible and suppress invalid percentage comparisons', async () => {
  const comparison = comparisonFixture();
  comparison.status = 'partial'; comparison.execution_status = 'timed_out';
  comparison.branches.candidate.status = 'failed';
  comparison.branches.candidate.seeds['17'].status = 'failed';
  comparison.branches.candidate.seeds['17'].training_stable = false;
  delete comparison.branches.candidate.seeds['17'].arms.clipped;
  comparison.failures = [{type: 'TimeoutError', phase: 'candidate/clipped'}];
  const h = await harness({initialRun: {id: 'partial', status: 'timed_out', comparison, events: []}});
  const area = h.get('comparison');
  const raw = area.querySelectorAll('table')[0];
  assert.equal(raw.querySelectorAll('tbody')[0].children.length, 4);
  assert.match(descendantText(raw), /missing arm/);
  assert.doesNotMatch(descendantText(raw), /\+10\.00%/);
  assert.match(descendantText(area), /Comparison status: partial/);
  assert.match(descendantText(area), /TimeoutError/);
  assert.match(descendantText(area), /failed recorded checks/);
});

test('actual worker progress is shown separately from planned episode counts', async () => {
  const h = await harness({initialRun: {id: 'working', status: 'running', events: [],
    progress: {completed_episodes: 18, maximum_episodes: 56, phase: 'baseline/seed-17/r1/uniform/train-1'}}});
  assert.equal(h.get('progress').hidden, false);
  assert.equal(h.get('progress-bar').max, 56);
  assert.equal(h.get('progress-bar').value, 18);
  assert.match(h.get('progress-label').textContent, /18 \/ 56 episodes completed/);
  assert.match(h.get('progress-label').textContent, /baseline\/seed-17\/r1\/uniform\/train-1/);
  assert.match(h.get('progress-label').textContent, /still in progress/);
  h.setState({...h.state(), run: {...h.state().run, progress: {maximum_episodes: 56}}});
  await h.fire('refresh');
  assert.equal(h.get('progress').hidden, true);
});

test('next-run proposal exposes its candidate and planned budget without launching or hiding diagnosis gaps', async () => {
  const comparison = comparisonFixture();
  comparison.next_run_proposal = {request: {schema: 'fusion-shadow-config-v1', change: 'td3_actor_lr', value: 0.00005, seeds: [29, 43], rounds: 1},
    budget: {total_episodes: 112, maximum_wall_seconds: 600}, reason: 'Repeat the same hypothesis across two additional paired seeds.',
    seed_selection_scope: 'Check the development and confirmation ledger before launch.', automatic_launch: false};
  const h = await harness({initialRun: {id: 'proposed', status: 'completed', events: [], comparison}});
  const proposal = h.get('comparison').children.find((item) => item.querySelectorAll('h4').some((heading) => heading.textContent === 'Proposed next run · not launched'));
  assert(proposal && proposal.tag === 'article');
  assert.match(descendantText(proposal), /TD3 actor learning rate = 0\.00005/);
  assert.match(descendantText(proposal), /Paired training seeds 29, 43/);
  assert.match(descendantText(proposal), /Rounds per arm 1/);
  assert.match(descendantText(proposal), /Planned episodes, both configurations 112/);
  assert.match(descendantText(proposal), /Maximum wall time 600 seconds/);
  assert.match(descendantText(proposal), /Repeat the same hypothesis/);
  assert.match(descendantText(proposal), /Check the development and confirmation ledger/);
  assert.equal(proposal.querySelectorAll('button').length, 0);
  comparison.next_run_proposal.request = null;
  comparison.next_run_proposal.reason = 'Training was incomplete; diagnose the saved failure first.';
  h.setState({...h.state(), run: {...h.state().run, comparison}}); await h.fire('refresh');
  assert.match(descendantText(h.get('comparison')), /No runnable configuration was proposed/);
  assert.match(descendantText(h.get('comparison')), /diagnose the saved failure first/);
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
});
