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

async function harness({available = true, initialRun, initialResearchRun, researchConfigured = false, researchCatalog, researchSources = {}} = {}) {
  const elements = {};
  for (const match of html.matchAll(/<([a-z][a-z0-9]*)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    assert(!elements[match[3]], `Duplicate id: ${match[3]}`);
    const element = new Element(match[1]);
    element.value = /\bvalue="([^"]*)"/.exec(match[2])?.[1] || '';
    elements[match[3]] = element;
  }
  const examples = [...html.matchAll(/data-research-example="([^"]+)"/g)].map((match) => {
    const button = new Element('button'); button.dataset.researchExample = match[1]; return button;
  });
  elements['research-shadow-rounds'].value = '1';
  const calls = [], timers = [], pendingResponses = [];
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
    URL, URLSearchParams, Set, JSON, AbortController,
    setTimeout: (fn, delay) => { const timer = {fn, delay, active: true}; timers.push(timer); return timer; },
    clearTimeout: (timer) => { if (timer) timer.active = false; },
    document: {getElementById: (id) => elements[id], createElement: (tag) => new Element(tag), querySelectorAll: () => examples},
    fetch: async (url, options = {}) => {
      calls.push({url, options});
      const override = pendingResponses.findIndex((item) => item.url === url && item.method === (options.method || 'GET'));
      if (override >= 0) {
        const selected = pendingResponses.splice(override, 1)[0];
        if (selected.response instanceof Error) throw selected.response;
        if (typeof selected.response === 'function') return selected.response(options);
        return selected.response;
      }
      if (url === '/api/research') return {ok: true, json: async () => research};
      if (url.startsWith('/api/research/source?')) {
        const source = researchSources[new URL(url, 'http://localhost').searchParams.get('source_id')];
        return {ok: Boolean(source), status: source ? 200 : 404, json: async () => source || {error: 'Unknown research source'}};
      }
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
    setResearchState: (value) => { research = {...research, ...value}; },
    respondOnce: (url, method, response) => pendingResponses.push({url, method, response})};
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

test('one focused TD3 audit is selected by default and all three saved scenarios stay visible', async () => {
  const ids = ['dc0df0a81413479e979a18895e1fe2fe', 'fbb83d8ee93a424497aeacc0a2633dc3', '12eddcebd2c548d889b41d3f2adace91'];
  const records = ids.map((id, index) => ({
    id: `examples/evidence/shadow/${id}/comparison.json`, title: 'Opaque original title',
    kind: 'experiment', status: index === 1 ? 'failed' : 'complete', adapter: 'shadow_comparison'
  }));
  const h = await harness({researchConfigured: true, researchCatalog: [
    {id: 'examples/evidence/shadow/993df979e388405297f0195bfca504d6/comparison.json', kind: 'experiment', status: 'complete'},
    ...records.slice().reverse()
  ]});
  const checked = h.elements['research-catalog'].querySelectorAll('input').filter((input) => input.checked);
  assert.deepEqual(checked.map((input) => input.value), [records[2].id]);
  assert.deepEqual(h.elements['research-catalog'].querySelectorAll('strong').slice(0, 3).map((item) => item.textContent), [
    'TD3 critic action coordinates: normalized → physical',
    'SAC learning rate: 0.0003 → 0.0001',
    'SAC critic federation: mixing 0.25 → 0.10'
  ]);
  const scope = h.elements['research-catalog'].querySelectorAll('small').filter((item) => /Saved SPARC-like TORAX simulation/.test(item.textContent));
  assert.equal(scope.length, 3);
  assert(scope.every((item) => /paired seed 29 · 1 round · 56 episodes · development, not confirmation/.test(item.textContent)));
  assert.equal(h.elements['research-selection-count'].textContent, '1/8 selected');
  assert.match(descendantText(h.elements['research-catalog']), /Recommended first audit.*not a promised federation win/);
  assert.equal(h.elements['research-allow-shadow'].checked, false);
  assert.equal(h.elements['research-live'].textContent, 'Analyze saved data + literature ↗');
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
  h.elements['research-question'].value = 'Analyze these saved TORAX datasets';
  h.elements['research-form'].listeners.submit({preventDefault() {}}); await flush();
  const request = h.calls.find((call) => call.options.method === 'POST');
  assert.deepEqual(JSON.parse(request.options.body).record_ids, [records[2].id]);
  assert.equal(JSON.parse(request.options.body).allow_shadow, false);
});

test('invalid featured data is never silently selected as a valid experiment', async () => {
  const invalid = 'examples/evidence/shadow/dc0df0a81413479e979a18895e1fe2fe/comparison.json';
  const valid = 'examples/evidence/shadow/fbb83d8ee93a424497aeacc0a2633dc3/comparison.json';
  const h = await harness({researchCatalog: [
    {id: invalid, kind: 'experiment', status: 'invalid'},
    {id: valid, kind: 'experiment', status: 'complete'}
  ]});
  const controls = h.elements['research-catalog'].querySelectorAll('input');
  assert.equal(controls.find((item) => item.value === invalid).checked, false);
  assert.equal(controls.find((item) => item.value === invalid).disabled, true);
  assert.deepEqual(controls.filter((item) => item.checked).map((item) => item.value), [valid]);
});

test('beta generation is visibly separate and a saved permission never opts in a new request', async () => {
  assert.equal((html.match(/Experimental \/ Beta — may fail/g) || []).length, 2);
  assert.match(html, /Default: analyze saved simulation data/);
  assert.match(html, /Optional manual generation, separate from saved-data research/);
  assert.match(html, /failures and partial outputs are retained/i);
  assert.doesNotMatch(html, /fresh seed 17/);
  const h = await harness({researchConfigured: true, initialResearchRun: {
    id: 'previous-beta-run', mode: 'live', allow_shadow: true, status: 'completed', events: []
  }});
  assert.equal(h.elements['research-allow-shadow'].checked, false);
  assert.match(h.elements['research-live'].textContent, /Analyze saved data/);
  assert.match(h.get('start').textContent, /Generate TORAX data \(Beta\)/);
  assert.match(h.get('runtime').textContent, /Experimental \/ Beta/);
  h.elements['research-allow-shadow'].checked = true;
  h.elements['research-allow-shadow'].listeners.change();
  assert.equal(h.elements['research-live'].textContent, 'Analyze + allow beta generation ↗');
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
    assert.match(h.get('error').textContent, /beta trial did not complete/);
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


const publicationFixture = () => ({
  schema: 'fusion-grounded-publication-v1', status: 'grounded',
  answer: 'Recorded seed 17 endpoint costs are available. This does not establish efficacy.',
  facts: [{id: 'F001', kind: 'recorded_measurement', text: 'Candidate raw endpoint cost = 0.09626584305007646 for seed 17.',
    evidence: [{source_id: 'D001', locator: '/candidate/raw_cost', quote: '0.09626584305007646'}], source_id: 'D001', source_sha256: 'abc123',
    provenance: {scope: 'one paired seed', adapter: 'shadow_comparison', trial_id: 'pilot'}, context: {seed: 17, arm: 'clipped', view: 'raw'}}],
  literature_excerpts: [{id: 'L001', kind: 'exact_excerpt', text: 'Source states: model updates have to be clipped before adding privacy noise.',
    evidence: [{source_id: 'W001', locator: 'chars:0:70', quote: 'model updates have to be clipped before adding privacy noise.'}], source_id: 'W001', source_sha256: 'paper123',
    provenance: {scope: 'verbatim excerpt, no transfer claim'}}],
  recommendation: {status: 'proposal_only', title: 'Repeat the paired comparison', request: {change: 'clip_norm_fraction', value: 0.001, seeds: [29, 43]},
    budget: {episodes: 112}, controls: ['local', 'reset_only', 'uniform', 'clipped'], limitations: ['Development seeds only'], evidence: [], provenance: {scope: 'bounded proposal'}},
  limitations: ['Only one training seed was recorded.'], scientific_efficacy_established: false, automatic_policy_changes_allowed: false,
});

test('legacy accepted model draft cannot become primary evidence', async () => {
  const h = await harness({initialResearchRun: {id: 'legacy', mode: 'live', status: 'completed', report: {
    answer: 'Federation always improves every reactor.', claims: [{id: 'C1', text: 'Legacy unsupported claim', kind: 'measured', confidence: 1,
      verification: {status: 'verified', reason: 'Old model approved'}, evidence: []}], judge: {decision: 'accept', reason: 'Old model approval'},
    next_runs: [{title: 'Automatically promote the candidate', hypothesis: 'Unverified claim'}], limitations: ['Old model limitation'], stop_reason: 'Everything is proven',
  }}});
  assert.match(h.elements['research-answer'].textContent, /No source-backed publication/);
  assert.doesNotMatch(descendantText(h.elements['research-claims']), /Legacy unsupported/);
  assert.doesNotMatch(descendantText(h.elements['research-proposals']), /Automatically promote/);
  assert.equal(h.elements['research-model-draft'].open, false);
  assert.match(h.elements['research-model-answer'].textContent, /Federation always/);
  assert.match(descendantText(h.elements['research-model-claims']), /Unverified model interpretation/);
  assert.match(descendantText(h.elements['research-model-proposals']), /MODEL PROPOSAL.*UNVERIFIED.*NOT LAUNCHED/);
  assert.match(descendantText(h.elements['research-judgment']), /Judge accepted the draft/);
  assert.match(h.elements['research-stop-reason'].textContent, /Publication: unavailable/);
});

test('grounded publication isolates server records and exact excerpts from model hypotheses', async () => {
  const publication = publicationFixture();
  const h = await harness({initialResearchRun: {id: 'grounded', mode: 'live', status: 'completed', report: {
    answer: 'Untrusted legacy answer should not be used', grounded_publication: publication,
    model_interpretation: {hypotheses: [{text: 'Clipping may improve transfer beyond this seed', evidence_ids: ['D001']}], gaps: ['More seeds are needed'], judge: {decision: 'accept'}},
    model_draft: {answer: 'Original draft retained', claims: [{id: 'C1', text: 'Original model claim', evidence: []}]},
  }}});
  assert.equal(h.elements['research-answer'].textContent, publication.answer);
  const facts = descendantText(h.elements['research-claims']);
  assert.match(facts, /Recorded measurement/);
  assert.match(facts, /raw endpoint cost = 0.09626584305007646/);
  assert.match(facts, /one paired seed/);
  assert.match(facts, /abc123/);
  assert.doesNotMatch(facts, /Clipping may improve|Original model claim|Confidence/);
  assert.match(descendantText(h.elements['research-literature']), /Exact source excerpt/);
  assert.equal(h.elements['research-literature'].querySelectorAll('blockquote').length, 1);
  assert.match(descendantText(h.elements['research-model-claims']), /Clipping may improve/);
  assert.match(descendantText(h.elements['research-model-claims']), /Original model claim/);
  assert.equal(h.elements['research-model-draft'].open, false);
  assert.match(descendantText(h.elements['research-limitations']), /Only one training seed/);
  assert.match(h.elements['research-stop-reason'].textContent, /Scientific efficacy is not established/);
  assert.match(h.elements['research-stop-reason'].textContent, /Automatic policy changes are disabled/);
});

test('insufficient evidence packet retains recorded measurements without implying proof', async () => {
  const publication = publicationFixture(); publication.status = 'insufficient_evidence'; publication.recommendation = null;
  const h = await harness({initialResearchRun: {id: 'partial', mode: 'live', report: {grounded_publication: publication}}});
  assert.match(descendantText(h.elements['research-claims']), /Recorded measurement/);
  assert.match(h.elements['research-stop-reason'].textContent, /Publication: insufficient evidence/);
  assert.match(descendantText(h.elements['research-proposals']), /No bounded next experiment proposal/);
});

for (const [name, change] of [
  ['unknown schema', {schema: 'future-schema'}], ['unknown status', {status: 'model_accepted'}],
  ['scientific proof assertion', {scientific_efficacy_established: true}], ['automatic policy assertion', {automatic_policy_changes_allowed: true}],
  ['missing safety contract', {automatic_policy_changes_allowed: undefined}], ['malformed facts', {facts: 'model text'}], ['null fact', {facts: [null]}],
  ['missing citation', {facts: [{text: 'Uncited number', evidence: []}]}], ['malformed citation', {facts: [{text: 'Claim', evidence: [{source_id: 'D1'}]}]}],
]) test(`invalid publication fails closed: ${name}`, async () => {
  const h = await harness({initialResearchRun: {id: 'invalid', mode: 'live', report: {answer: 'Model says accepted', grounded_publication: {...publicationFixture(), ...change}}}});
  assert.match(h.elements['research-answer'].textContent, /No source-backed publication/);
  assert.doesNotMatch(descendantText(h.elements['research-claims']), /0.09626584305007646/);
});

test('bounded proposal is read-only and does not inherit model acceptance', async () => {
  const publication = publicationFixture();
  const h = await harness({initialResearchRun: {id: 'proposal', mode: 'live', report: {grounded_publication: publication}}});
  const proposal = h.elements['research-proposals'];
  assert.match(descendantText(proposal), /RECORDED PROPOSAL.*NOT LAUNCHED/);
  assert.match(descendantText(proposal), /clip_norm_fraction/);
  assert.match(descendantText(proposal), /112/);
  assert.equal(proposal.querySelectorAll('button').length, 0);
  assert.equal(proposal.querySelectorAll('input').length, 0);
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
  publication.recommendation.status = 'approved';
  h.setResearchState({run: {id: 'proposal', mode: 'live', report: {grounded_publication: publication}}});
  h.elements['research-refresh'].listeners.click(); await flush();
  assert.match(descendantText(proposal), /No bounded next experiment proposal/);
});

test('published evidence opens exact saved source with locator and hash', async () => {
  const publication = publicationFixture();
  const h = await harness({initialResearchRun: {id: 'source-run', mode: 'live', report: {grounded_publication: publication}}, researchSources: {
    D001: {id: 'D001', kind: 'experiment', title: 'Saved paired comparison', sha256: 'abc123', content: '{"candidate":{"raw_cost":0.09626584305007646}}', uri: 'javascript:alert(1)'},
  }});
  h.elements['research-claims'].querySelectorAll('button')[0].listeners.click(); await flush();
  assert.equal(h.elements['research-source-dialog'].open, true);
  assert.equal(h.elements['research-source-title'].textContent, 'Saved paired comparison');
  const sourceText = descendantText(h.elements['research-source-body']);
  assert.match(sourceText, /0.09626584305007646/);
  assert.match(sourceText, /Locator: \/candidate\/raw_cost/);
  assert.match(sourceText, /SHA-256: abc123/);
  assert.match(sourceText, /JSON value at the recorded pointer/);
  assert.match(sourceText, /matching text alone would not verify the claim/);
  assert.equal(h.elements['research-source-body'].querySelectorAll('a').length, 0);
  assert(h.calls.some((call) => call.url === '/api/research/source?run_id=source-run&source_id=D001'));
});

test('refresh to a historical report clears prior publication and never executes HTML', async () => {
  const publication = publicationFixture(); publication.facts[0].text = '<script>alert(1)</script>';
  const h = await harness({initialResearchRun: {id: 'first', mode: 'live', report: {grounded_publication: publication}}});
  assert.match(descendantText(h.elements['research-claims']), /<script>alert\(1\)<\/script>/);
  h.setResearchState({run: {id: 'second', mode: 'live', report: {answer: 'Historical draft', judge: {decision: 'accept'}}}});
  h.elements['research-refresh'].listeners.click(); await flush();
  assert.match(h.elements['research-answer'].textContent, /No source-backed publication/);
  assert.doesNotMatch(descendantText(h.elements['research-claims']), /alert|0.0962/);
  assert.match(descendantText(h.elements['research-literature']), /No exact literature excerpts/);
  assert.equal(h.elements['research-model-answer'].textContent, 'Historical draft');
});


test('fact cards visibly distinguish current and historical trial provenance', async () => {
  const publication = publicationFixture();
  publication.facts[0].source_label = 'Current research trial';
  publication.facts[0].provenance.origin = 'research_shadow_execution';
  publication.facts.push({...publication.facts[0], id: 'F002', source_label: undefined, provenance: {origin: 'saved_shadow_snapshot', trial_id: 'older17'}});
  const h = await harness({initialResearchRun: {id: 'mixed', mode: 'live', report: {grounded_publication: publication}}});
  const cards = h.elements['research-claims'].children;
  assert.match(cards[0].children.find((item) => item.className === 'research-fact-scope').textContent, /Current research trial.*trial pilot/);
  assert.match(cards[1].children.find((item) => item.className === 'research-fact-scope').textContent, /Saved trial.*trial older17/);
});

test('proposal preserves its reason, evaluation-subset explanation and unverified seed scope', async () => {
  const publication = publicationFixture();
  publication.recommendation.text = 'Repeat the frozen hypothesis; this is not efficacy evidence.';
  publication.recommendation.budget_scope = 'Gate, initial and report episodes are subsets of evaluation, not additional episodes.';
  publication.recommendation.seed_selection_scope = 'Check the full development and confirmation ledger before choosing these seeds.';
  const h = await harness({initialResearchRun: {id: 'scopes', mode: 'live', report: {grounded_publication: publication}}});
  const text = descendantText(h.elements['research-proposals']);
  assert.match(text, /Repeat the frozen hypothesis/);
  assert.match(text, /subsets of evaluation, not additional episodes/);
  assert.match(text, /Check the full development and confirmation ledger/);
});

test('missing request is a diagnosis, and rejected hypotheses stay visibly rejected', async () => {
  const publication = publicationFixture();
  publication.recommendation.request = null; publication.recommendation.budget = null;
  publication.recommendation.text = 'Diagnose the failed trial before a larger comparison.';
  const h = await harness({initialResearchRun: {id: 'diagnosis', mode: 'live', report: {grounded_publication: publication,
    model_interpretation: {hypotheses: [{text: 'Unjustified mechanism', status: 'critic_rejected', evidence_ids: []}], judge: {decision: 'accept'}}}}});
  assert.match(descendantText(h.elements['research-proposals']), /Diagnosis before another run/);
  assert.match(descendantText(h.elements['research-proposals']), /Diagnose the failed trial/);
  assert.doesNotMatch(descendantText(h.elements['research-proposals']), /112/);
  assert.match(descendantText(h.elements['research-model-claims']), /Critic rejected this hypothesis/);
  assert.equal(h.elements['research-model-draft'].open, false);
});

test('hypotheses with unknown sources are visibly excluded from recorded findings', async () => {
  const h = await harness({initialResearchRun: {id: 'quarantine', mode: 'live', report: {
    grounded_publication: publicationFixture(), model_interpretation: {
      hypotheses: [], rejected_hypotheses: [{text: 'Invalid speculative statement',
        status: 'invalid_source_references', evidence_ids: [], rejected_evidence_ids: ['made-up']}]
    }}}});
  assert.match(descendantText(h.elements['research-model-claims']), /Excluded: invalid source references/);
  assert.doesNotMatch(descendantText(h.elements['research-claims']), /Invalid speculative statement/);
  assert.equal(h.elements['research-model-draft'].open, false);
});

test('the visible app loads only the new research workspace and keeps legacy surfaces inert', () => {
  assert.deepEqual([...html.matchAll(/<script src="([^"]+)"/g)].map((item) => item[1]), ['/research.js']);
  assert.match(html, /id="legacy-workspace" hidden inert aria-hidden="true"/);
  const visible = html.slice(0, html.indexOf('<div id="legacy-workspace"'));
  assert.doesNotMatch(visible, /href="#(?:investigate|thermal|physics|assembly|report)"/);
  assert.doesNotMatch(visible, /TORAX \/ FLOWER|Flower hosted|SuperGrid/);
  assert.match(visible, /TORAX \/ NEBIUS \/ TAVILY \/ LANGGRAPH/);
});

test('a failed new run retains the previous report and trace with explicit separate run labels', async () => {
  const h = await harness({researchConfigured: true, initialResearchRun: {
    id: 'previous-good', mode: 'live', status: 'completed', question: 'Previous question',
    events: [{agent: 'planner', kind: 'plan', message: 'Prior recorded decision'}],
    report: {grounded_publication: publicationFixture()}
  }});
  h.setResearchState({run: {id: 'new-failed', mode: 'live', status: 'failed', events: [], error: 'RAW SECRET PROVIDER TRACE'}});
  h.elements['research-refresh'].listeners.click(); await flush();
  assert.equal(h.elements['research-results'].hidden, false);
  assert.match(h.elements['research-answer'].textContent, /Recorded seed 17 endpoint/);
  assert.match(h.elements['research-retained-evidence'].textContent, /Previous saved report · run previous-good.*current run new-failed/);
  assert.equal(h.elements['research-retained-evidence'].hidden, false);
  assert.match(h.elements['research-trace-context'].textContent, /Previous trace · run previous-good.*current run new-failed/);
  assert.match(descendantText(h.elements['research-trace']), /Prior recorded decision/);
  assert.doesNotMatch(descendantText(h.elements['research-error']), /RAW SECRET/);
  assert.match(h.elements['research-error-message'].textContent, /stopped before completing/);
  h.elements['research-refresh'].listeners.click(); await flush();
  assert.equal(h.elements['research-retained-evidence'].hidden, false);
  const citation = h.elements['research-claims'].querySelectorAll('button')[0];
  citation.listeners.click(); await flush();
  assert(h.calls.some((call) => call.url.includes('run_id=previous-good')));
});

test('explicit recovery checks saved state then retries without authorizing physics', async () => {
  const h = await harness({researchConfigured: true, initialResearchRun: {id: 'failed1', mode: 'live', status: 'failed', events: [], error: 'No provider response'}});
  h.elements['research-question'].value = 'Read the saved experiment';
  h.elements['research-allow-shadow'].checked = true;
  const before = h.calls.length;
  h.elements['research-retry'].listeners.click(); await flush();
  const retryCalls = h.calls.slice(before);
  assert.equal(retryCalls[0].url, '/api/research');
  assert.equal(retryCalls[0].options.method, undefined);
  const posts = retryCalls.filter((call) => call.options.method === 'POST');
  assert.equal(posts.length, 1);
  assert.equal(posts[0].url, '/api/research');
  assert.equal(JSON.parse(posts[0].options.body).allow_shadow, false);
  assert.equal(h.elements['research-allow-shadow'].checked, false);
  assert.equal(h.calls.filter((call) => call.url === '/api/shadow/start').length, 0);
});

test('recovery never duplicates an accepted request discovered after a lost POST response', async () => {
  const h = await harness({researchConfigured: true});
  h.elements['research-question'].value = 'Read saved evidence';
  h.respondOnce('/api/research', 'POST', () => {
    h.setResearchState({active: true, run: {id: 'accepted-on-server', mode: 'live', status: 'running', events: []}});
    throw new Error('PRIVATE TRANSPORT TOKEN');
  });
  h.elements['research-form'].listeners.submit({preventDefault() {}}); await flush();
  assert.match(h.elements['research-error-message'].textContent, /saved server state.*no duplicate request/);
  assert.match(h.elements['research-run-meta'].textContent, /accepted-on-server/);
  assert.equal(h.elements['research-retry'].disabled, true);
  h.elements['research-retry'].listeners.click(); await flush();
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 1);
  assert.doesNotMatch(h.elements['research-error-message'].textContent, /PRIVATE/);
});

test('a timed-out read restores controls and preserves visible evidence without exposing exception text', async () => {
  const h = await harness({researchConfigured: true, initialResearchRun: {id: 'known', mode: 'live', status: 'insufficient_evidence', report: {grounded_publication: publicationFixture()}, events: [{kind: 'read', message: 'Saved trace'}]}});
  h.respondOnce('/api/research', 'GET', (options) => new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(new Error('INTERNAL SECRET')))));
  h.elements['research-refresh'].listeners.click(); await flush();
  const deadline = h.timers.find((timer) => timer.active && timer.delay === 20000);
  assert(deadline); deadline.fn(); await flush();
  assert.match(h.elements['research-error-message'].textContent, /within 20 seconds/);
  assert.equal(h.elements['research-refresh'].disabled, false);
  assert.equal(h.elements['research-live'].disabled, false);
  assert.equal(h.elements['research-results'].hidden, false);
  assert.match(descendantText(h.elements['research-trace']), /Saved trace/);
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
});

test('backend errors are safe public categories and local recovery remains available without model calls', async () => {
  const h = await harness({researchConfigured: true, initialResearchRun: {id: 'failed', mode: 'live', status: 'failed', events: [], error: 'Secret exception'}});
  h.elements['research-question'].value = 'Read saved data';
  h.respondOnce('/api/research', 'POST', {ok: false, status: 429, json: async () => ({error: 'API key = DO_NOT_DISPLAY'})});
  h.elements['research-form'].listeners.submit({preventDefault() {}}); await flush();
  assert.match(h.elements['research-error-message'].textContent, /provider limit.*HTTP 429/);
  assert.doesNotMatch(h.elements['research-error-message'].textContent, /DO_NOT_DISPLAY/);
  h.elements['research-allow-shadow'].checked = true;
  h.elements['research-error-offline'].listeners.click(); await flush();
  const body = JSON.parse(h.calls.filter((call) => call.options.method === 'POST').at(-1).options.body);
  assert.equal(body.mode, 'offline'); assert.equal(body.allow_shadow, false);
});

test('manual recovery rechecks active runs and never retries a successfully completed run', async () => {
  const h = await harness({researchConfigured: true, initialResearchRun: {id: 'old-failure', status: 'failed', error: 'Failure'}});
  h.setResearchState({active: true, run: {id: 'already-running', mode: 'live', status: 'running', events: []}});
  h.elements['research-retry'].listeners.click(); await flush();
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
  assert.equal(h.elements['research-retry'].disabled, true);
  h.setResearchState({active: false, run: {id: 'already-running', mode: 'live', status: 'completed', report: {grounded_publication: publicationFixture()}}});
  h.elements['research-refresh'].listeners.click(); await flush();
  h.elements['research-retry'].listeners.click(); await flush();
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
});

test('a lost beta start response reconciles state without repeating simulation start', async () => {
  const h = await harness(); await h.fire('form', 'submit');
  h.respondOnce('/api/shadow/start', 'POST', () => {
    h.setState({...h.state(), active: true, run: {...h.state().run, status: 'running'}});
    throw new Error('PRIVATE WORKER CONNECTION');
  });
  await h.fire('start');
  assert.equal(h.get('start').disabled, true);
  assert.match(h.get('error').textContent, /no simulation was restarted automatically/);
  assert.doesNotMatch(h.get('error').textContent, /PRIVATE/);
  assert.equal(h.calls.filter((call) => call.url === '/api/shadow/start').length, 1);
});

test('terminal persistence failure gives safe storage recovery guidance and keeps current export available', async () => {
  const h = await harness({researchConfigured: true, initialResearchRun: {
    id: 'not-saved-yet', mode: 'live', status: 'failed', error: 'PRIVATE DISK TRACE',
    persistence_failure: {terminal_state_persistence_confirmed: false, error: 'PRIVATE FILE PATH'},
    events: [{kind: 'persistence_failure', message: 'Terminal save was not confirmed'}],
    report: {grounded_publication: publicationFixture()}
  }});
  const message = h.elements['research-error-message'].textContent;
  assert.match(message, /could not be saved to disk/);
  assert.match(message, /Download the current run.*before restarting the server or starting another run/);
  assert.match(message, /available storage and write permissions/);
  assert.doesNotMatch(message, /PRIVATE/);
  assert.equal(h.elements['research-export'].hidden, false);
  assert.equal(h.elements['research-export'].href, '/api/research/export?run_id=not-saved-yet');
  assert.equal(h.elements['research-results'].hidden, false);
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
});

test('grounded local inspection shows saved measurements and intentionally skips literature and model review', async () => {
  const publication = publicationFixture();
  publication.status = 'insufficient_evidence'; publication.literature_excerpts = [];
  publication.facts = Array.from({length: 12}, (_, index) => ({...publication.facts[0], id: `F${index + 1}`}));
  publication.limitations = ['Local inspection only; no literature search or new training.'];
  const h = await harness({initialResearchRun: {
    id: 'local-audit', mode: 'offline', status: 'inspection_complete', events: [{kind: 'inspection_complete', message: 'Read saved comparison'}],
    report: {grounded_publication: publication, evidence_sufficient: false, orchestration: {status: 'inspection_complete'}}
  }});
  assert.equal(h.elements['research-result-mode'].textContent, 'Local inspection — saved measurements only');
  assert.equal(h.elements['research-claims'].children.length, 12);
  assert.match(h.elements['research-answer'].textContent, /Recorded seed 17 endpoint/);
  assert.doesNotMatch(h.elements['research-answer'].textContent, /No source-backed publication/);
  assert.match(descendantText(h.elements['research-literature']), /No literature was retrieved.*saved measurements only/);
  assert.match(descendantText(h.elements['research-proposals']), /Repeat the paired comparison/);
  assert.match(h.elements['research-stop-reason'].textContent, /No LLM, Tavily retrieval, model judge or new training ran/);
  assert.equal(h.elements['research-model-draft'].hidden, true);
  assert.equal(h.elements['research-agent-map'].hidden, true);
  assert.equal(h.calls.filter((call) => call.options.method === 'POST').length, 0);
  h.setResearchState({run: {id: 'legacy-offline', mode: 'offline', status: 'inspection_complete', report: {answer: 'Legacy draft'}}});
  h.elements['research-refresh'].listeners.click(); await flush();
  assert.match(h.elements['research-answer'].textContent, /No source-backed publication/);
  assert.equal(h.elements['research-model-draft'].hidden, false);
  assert.match(h.elements['research-model-answer'].textContent, /Legacy draft/);
});
