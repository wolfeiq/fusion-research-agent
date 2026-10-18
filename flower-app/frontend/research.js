'use strict';

// Separate research state from the Flower run. Never execute report HTML or URLs.
(() => {
  const byId = (id) => document.getElementById(`research-${id}`);
  if (!byId('form')) return;
  const examples = {
    coldstart: 'Do these saved TORAX experiments support federation helping a cold-start reactor? Read the selected experiment evidence first. Compare local, reset-only, uniform and clipped outcomes, separate raw learners from gate fallbacks, and distinguish endpoint cost from shots-to-competence. Cite exact saved values and extracted primary sources. Inspect a federation dossier only if one is selected, and propose a small fair next experiment. Do not change existing runs.',
    aggregation: 'Which aggregation policy should we test next? Read the selected experiment evidence first and check whether clipping actually changed the recorded updates and outcomes. Cite exact values, critic roles, paired controls and limitations. Use inspect_federation, compare_clipping and draft_policy_trial only if a selected federation dossier supports those tools; a saved shadow comparison is not a dossier. Compare primary literature and propose a bounded next run with a budget and stopping rule. Do not launch or alter a run.',
    td3: 'What does the selected experiment evidence show about TD3 heating stability, and could local learning explain apparent federation benefit or harm? Read the recorded learning-rate comparisons, repair controls and heating diagnostics that are actually available. Cite exact values and extracted primary literature. Inspect a federation dossier only if one is selected. Separate measured outcomes from explanations that need another experiment; propose a small fair test without changing existing runs.',
    sac: 'How well does SAC deposition control cooperate with TD3 heating, and where might federation interfere? Read the selected experiment evidence first, keeping the SAC and TD3 roles and their critic branches separate. Cite exact recorded clipping, shared power-budget and combined-controller findings where available, alongside extracted primary literature. Inspect a federation dossier only if one is selected. Report missing evidence and propose a controlled next test without claiming online optimization or launching training.',
    transfer: 'Why could critic federation hurt TD3 and SAC control in our TORAX experiments? Read the selected experiment evidence, cite exact recorded outcomes and failures, and compare extracted primary research. Distinguish negative transfer from unstable local learning, keeping raw and gated results separate. Inspect a federation dossier only if one is selected, and propose a small controlled next experiment.',
    architecture: 'Audit our federated TD3 heating and SAC deposition architecture against the selected experiment evidence and extracted primary research. Cite exact sources for what is implemented and measured; identify missing evidence before a larger run. Distinguish in-process shadow aggregation from actual Flower transport. Inspect a federation dossier only if one is selected.'
  };
  let snapshot = null;
  let timer = null;
  let requesting = false;
  let submitting = false;
  let selected = new Set();
  let catalogSeen = false;
  let catalogSignature = '';
  let traceSignature = '';
  let reportSignature = '';
  let sourceRequest = 0;
  let watchShadow = null;
  const node = (tag, className, text) => {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined && text !== null) result.textContent = String(text);
    return result;
  };
  const display = (value) => typeof value === 'string' ? value : value == null ? '' : JSON.stringify(value, null, 2);
  const array = (value) => Array.isArray(value) ? value : [];
  const label = (value) => String(value ?? 'unknown').replaceAll('_', ' ');
  const roles = {
    planner: 'Planner', td3_specialist: 'TD3 specialist', sac_specialist: 'SAC specialist', federation_strategist: 'Federation strategist',
    literature_researcher: 'Literature researcher', writer: 'Writer', judge: 'Judge'
  };
  function validSelection() {
    return selected.size > 0 && selected.size <= 8 && array(snapshot?.catalog).some((record) => selected.has(String(record.id)) && record.kind === 'experiment' && record.status !== 'invalid');
  }
  function webURL(value) {
    try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : null; }
    catch { return null; }
  }
  function endpoint(path, values) { return `/api/research/${path}?${new URLSearchParams(values)}`; }
  async function request(path, body) {
    const response = await fetch(path, body === undefined ? {cache: 'no-store'} : {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-Fusion-UI': '1'}, body: JSON.stringify(body)
    });
    let data;
    try { data = await response.json(); } catch { throw new Error(`Research service returned an unreadable response (${response.status}).`); }
    if (!response.ok) throw new Error(data.error || `Research request failed (${response.status}).`);
    return data;
  }
  function showError(message) { byId('error').textContent = message || ''; byId('error').hidden = !message; }
  function selectionChanged() {
    byId('selection-count').textContent = `${selected.size}/8 selected${selected.size > 8 ? ' · select at most 8' : selected.size && !validSelection() ? ' · include an experiment' : ''}`;
    controls();
  }
  function controls() {
    const active = Boolean(snapshot?.active || submitting);
    const configured = snapshot?.configured || {};
    byId('live').disabled = active || requesting || !validSelection() || !configured.nebius || !configured.tavily;
    byId('offline').disabled = active || requesting || !validSelection();
    byId('live').textContent = active && snapshot?.run?.mode === 'live' ? 'Research in progress…' : 'Analyze with Nebius + Tavily ↗';
    byId('offline').textContent = active && snapshot?.run?.mode === 'offline' ? 'Inspecting evidence…' : 'Inspect local evidence';
    byId('live').title = !configured.nebius || !configured.tavily ? 'Configure NEBIUS_API_KEY and TAVILY_API_KEY on the server to run live research.' : '';
    byId('question').disabled = active;
    byId('allow-shadow').disabled = active;
    byId('catalog').querySelectorAll('input').forEach((input) => { input.disabled = active || input.dataset.invalid === 'true' || (!input.checked && selected.size >= 8); });
    document.querySelectorAll('[data-research-example]').forEach((button) => { button.disabled = active; });
  }
  function renderCatalog(records) {
    const signature = JSON.stringify(records);
    if (signature === catalogSignature) return;
    catalogSignature = signature;
    const usable = records.filter((record) => record.status !== 'invalid');
    const valid = new Set(usable.map((record) => String(record.id)));
    if (!catalogSeen && usable.length) {
      // Keep the failed-transfer evidence alongside stability work and its protocol.
      // These are visible choices, not a favorable-results filter.
      const previousDefaults = [
        usable.find((record) => record.kind === 'experiment' && /coldstart_kaggle_audit\/.*\/audit_summary\.json$/i.test(record.id)) || usable.find((record) => record.kind === 'experiment' && /coldstart.*(audit|round2|smoke)/i.test(record.id)),
        usable.find((record) => record.kind === 'experiment' && /heating.*coupled_comparison/i.test(record.id)) || usable.find((record) => record.kind === 'experiment' && /heating.*comparison/i.test(record.id)),
        usable.find((record) => record.kind === 'protocol' && /coldstart_coupled_federation/i.test(record.id))
      ].filter(Boolean);
      const federationDefaults = ['federation:local-smoke', 'federation:kaggle-359552340'].map((id) => usable.find((record) => record.id === id)).filter(Boolean);
      const publishedDefaults = usable.filter((record) => record.kind === 'experiment' && /^examples\/evidence\/shadow\/[a-f0-9]{32}\/comparison\.json$/.test(String(record.id))).slice(0, 2);
      const repair = usable.find((record) => /(?:^|\/)repair_comparison\.json$/i.test(record.id));
      const chosen = publishedDefaults.length ? publishedDefaults : federationDefaults.length
        ? [...federationDefaults, ...(repair ? [repair] : []), ...previousDefaults].slice(0, 3)
        : previousDefaults;
      selected = new Set(chosen.map((record) => String(record.id)));
      for (const record of publishedDefaults.length ? [] : [...usable].sort((a, b) => Number(b.kind === 'experiment') - Number(a.kind === 'experiment'))) {
        if (selected.size >= 3) break;
        selected.add(String(record.id));
      }
      catalogSeen = true;
    } else selected = new Set([...selected].filter((id) => valid.has(id)));
    const items = records.map((record) => {
      const row = node('label', 'research-record');
      const checkbox = node('input'); checkbox.type = 'checkbox'; checkbox.value = String(record.id); checkbox.checked = selected.has(checkbox.value);
      checkbox.dataset.invalid = String(record.status === 'invalid');
      checkbox.addEventListener('change', () => { checkbox.checked ? selected.add(checkbox.value) : selected.delete(checkbox.value); selectionChanged(); });
      const description = node('span');
      description.append(node('strong', '', record.title || record.id), node('small', '', `${label(record.kind)} · ${label(record.status)}`));
      if (record.metadata?.adapter === 'federation_dossier' || String(record.id).startsWith('federation:')) description.append(node('span', 'research-telemetry-badge', 'Saved coordinator telemetry'));
      if (record.metadata?.adapter === 'shadow_comparison' || record.adapter === 'shadow_comparison') description.append(node('span', 'research-telemetry-badge', 'Recorded TORAX shadow pilot'));
      if (record.path) description.append(node('small', '', record.path));
      if (record.error) description.append(node('small', '', record.error));
      row.append(checkbox, description); return row;
    });
    byId('catalog').replaceChildren(...(items.length ? items : [node('p', 'muted', 'No saved records are available. Add an approved experiment summary before starting research.')]));
    selectionChanged();
  }
  function renderTrace(events, run) {
    const offline = run.id && run.mode === 'offline';
    byId('agent-map').hidden = Boolean(offline);
    byId('workflow-note').textContent = offline
      ? 'Deterministic local inspection. No agents, model calls, web searches or judge ran.'
      : 'Actual LangGraph roles. Highlighted roles have recorded events; event counts do not imply a role is currently running.';
    for (const [role, title] of Object.entries(roles)) {
      const count = offline ? 0 : events.filter((event) => event.agent === role).length;
      const card = byId(`role-${role}`);
      card.className = `research-role${role === 'planner' ? ' research-role-planner' : ''}${count ? ' research-role-seen' : ''}`;
      card.replaceChildren(node('span', '', `0${Object.keys(roles).indexOf(role) + 1}`), node('strong', '', title), node('small', '', count ? `${count} recorded event${count === 1 ? '' : 's'}` : 'No recorded events'));
    }
    const signature = JSON.stringify({events, mode: run.mode});
    if (signature === traceSignature) return;
    traceSignature = signature;
    const trace = byId('trace');
    const nearBottom = trace.scrollHeight - trace.scrollTop - trace.clientHeight < 80;
    if (!events.length) {
      const empty = node('div', 'research-empty'); empty.append(node('span', '', '01 → 07'), node('p', '', 'The actual decisions, evidence gaps and stopping reason will appear here.'));
      trace.replaceChildren(empty); return;
    }
    trace.replaceChildren(...events.map((event, index) => {
      const entry = node('article', 'research-trace-event');
      const agent = !offline && event.agent ? `${roles[event.agent] || label(event.agent)} · ` : '';
      const round = event.round !== undefined ? ` · round ${event.round}` : event.revision !== undefined ? ` · revision ${event.revision}` : event.iteration !== undefined ? ` · iteration ${event.iteration}` : '';
      entry.append(node('h4', '', `${String(index + 1).padStart(2, '0')} · ${agent}${label(event.kind || event.type || event.event || event.stage || 'decision')}${round}`));
      if (event.tool) entry.append(node('p', 'research-trace-tool', `Tool · ${display(event.tool)}`));
      const content = event.message || event.reason || event.detail || event.summary || event.action || event.data;
      if (content) entry.append(node('p', '', display(content)));
      const details = node('details'); details.append(node('summary', '', 'Inspect trace event'), node('pre', '', display(event))); entry.append(details);
      return entry;
    }));
    if (nearBottom) trace.scrollTop = trace.scrollHeight;
  }
  function citationButton(evidence, runId, title) {
    const button = node('button', 'research-citation', title || `[${evidence.source_id}] View source ↗`);
    button.type = 'button'; button.addEventListener('click', () => openSource(runId, evidence)); return button;
  }
  function confidence(value) {
    if (typeof value === 'number' && Number.isFinite(value)) return value >= 0 && value <= 1 ? `${Math.round(value * 100)}%` : String(value);
    return label(value || 'not rated');
  }
  function renderJudgment(report, mode) {
    const panel = byId('judgment');
    if (mode !== 'live') {
      panel.replaceChildren(node('h3', '', 'Local inspection · no model judge'), node('p', '', 'This mode checks saved records and citation locations deterministically. It does not produce a semantic research judgment.'));
      return;
    }
    const judge = report.judge;
    const titles = {accept: 'Judge accepted the draft', revise: 'Judge requested revision', insufficient: 'Judge found insufficient evidence'};
    const header = node('div', 'panel-title');
    header.append(node('h3', '', judge ? titles[judge.decision] || `Judge: ${label(judge.decision)}` : 'No graph judge verdict recorded'), node('span', 'pill', 'Model assessment · fallible'));
    panel.replaceChildren(header);
    panel.append(node('p', '', 'The judge tests whether the cited evidence supports the conclusions and proposed controls. Acceptance is not independent scientific validation or permission to run an experiment.'));
    const rubricFields = [['citation_support', 'Citation support'], ['experimental_fairness', 'Experimental fairness'], ['actionability', 'Actionability'], ['scope_honesty', 'Scope honesty']];
    const scores = rubricFields.filter(([key]) => Number.isInteger(judge?.rubric?.[key]) && judge.rubric[key] >= 0 && judge.rubric[key] <= 2);
    if (scores.length) {
      const rubric = node('div', 'research-judge-rubric');
      for (const [key, title] of scores) {
        const item = node('div', 'research-rubric-score');
        item.append(node('span', '', title), node('strong', '', `${judge.rubric[key]} / 2`));
        rubric.append(item);
      }
      panel.append(rubric, node('p', 'research-caption', 'Judge rubric: 0 = unmet · 1 = partial · 2 = met. Scores are recorded model assessments, not calibrated probabilities.'));
    }
    const verification = report.verification || {};
    const unsupported = array(report.claims).filter((claim) => claim.verification?.status === 'unsupported' || (claim.review || claim.semantic_review)?.verdict === 'unsupported');
    panel.append(node('p', 'research-judge-counts', `${verification.traceable_claims ?? array(report.claims).filter((claim) => claim.verification?.status === 'traceable').length} traceable claims · ${verification.unsupported_claims ?? unsupported.length} unsupported claims`));
    if (report.semantic_review) panel.append(node('p', 'research-caption', `Semantic review: ${label(display(report.semantic_review))}`));
    if (judge?.reason) panel.append(node('p', '', display(judge.reason)));
    const gaps = array(judge?.gaps);
    if (gaps.length) {
      panel.append(node('h4', '', 'Unresolved evidence gaps'));
      const list = node('ul'); list.append(...gaps.map((gap) => node('li', '', display(gap)))); panel.append(list);
    }
    if (unsupported.length) {
      panel.append(node('h4', '', 'Unsupported claims retained for inspection'));
      const list = node('ul');
      list.append(...unsupported.map((claim) => node('li', '', `${claim.id}: ${claim.text}`))); panel.append(list);
    }
    if (Array.isArray(judge?.unsupported_claims) && judge.unsupported_claims.length) {
      const list = node('ul'); list.append(...judge.unsupported_claims.map((claim) => node('li', '', display(claim)))); panel.append(list);
    }
  }
  function publicationFor(report) {
    const value = report.grounded_publication;
    // Only this server-owned protocol can enter the primary evidence view.
    const cardValid = (card) => card && typeof card === 'object' && typeof card.text === 'string'
      && Array.isArray(card.evidence) && card.evidence.length > 0
      && card.evidence.every((item) => item && typeof item.source_id === 'string' && typeof item.quote === 'string' && typeof item.locator === 'string');
    if (!value || value.schema !== 'fusion-grounded-publication-v1'
      || !['grounded', 'insufficient_evidence'].includes(value.status)
      || !Array.isArray(value.facts) || !Array.isArray(value.literature_excerpts)
      || !value.facts.every(cardValid) || !value.literature_excerpts.every(cardValid)
      || !Array.isArray(value.limitations) || !value.limitations.every((item) => typeof item === 'string') || typeof value.answer !== 'string'
      || value.scientific_efficacy_established !== false || value.automatic_policy_changes_allowed !== false) return null;
    return value;
  }
  function evidenceLinks(items, runId) {
    const citations = node('div', 'research-citations');
    array(items).forEach((item) => { if (item?.source_id) citations.append(citationButton(item, runId)); });
    if (!citations.childNodes.length) citations.append(node('span', 'research-caption', 'No supporting source attached.'));
    return citations;
  }
  function publicationCard(fact, runId, excerpt = false) {
    const card = node('article', `research-claim${excerpt ? ' research-excerpt' : ''}`);
    const badges = node('div', 'research-badges');
    badges.append(node('span', 'research-badge', fact.id || 'record'), node('span', 'research-badge supported', excerpt ? 'Exact source excerpt' : 'Recorded measurement'));
    const origin = fact.provenance?.origin;
    const scopeLabel = fact.source_label || (origin === 'research_shadow_execution' ? 'Current research trial'
      : origin === 'saved_shadow_snapshot' ? 'Saved trial' : excerpt ? 'Captured literature source' : 'Captured source');
    const trialLabel = fact.provenance?.trial_id ? ` · trial ${fact.provenance.trial_id}` : '';
    card.append(badges, node('p', 'research-fact-scope', `${scopeLabel}${trialLabel}`), node(excerpt ? 'blockquote' : 'h4', '', fact.text), evidenceLinks(fact.evidence, runId));
    if (fact.source_id) card.append(node('p', 'research-caption', `Source: ${fact.source_id}`));
    const details = node('details', 'research-fact-provenance');
    details.append(node('summary', '', 'Inspect measurement provenance'), node('pre', '', display({
      source_sha256: fact.source_sha256, provenance: fact.provenance, ...(fact.context ? {context: fact.context} : {})
    })));
    card.append(details); return card;
  }
  function modelClaimCard(claim, runId) {
    const card = node('article', 'research-claim');
    const badges = node('div', 'research-badges');
    badges.append(node('span', 'research-badge', claim.id || 'hypothesis'), node('span', 'research-badge uncertain', 'Unverified model interpretation'));
    if (claim.status === 'critic_rejected') badges.append(node('span', 'research-badge uncertain', 'Critic rejected this hypothesis'));
    if (claim.status === 'invalid_source_references') badges.append(node('span', 'research-badge uncertain', 'Excluded: invalid source references'));
    card.append(badges, node('h4', '', claim.text));
    if (claim.confidence !== undefined) card.append(node('p', 'research-caption', `Model confidence: ${confidence(claim.confidence)}. This is not a calibrated probability.`));
    if (claim.verification) card.append(node('p', 'research-verification', `Recorded citation check: ${label(claim.verification.status)}. ${claim.verification.reason || ''}`));
    const review = claim.review || claim.semantic_review;
    if (review) card.append(node('p', 'research-verification', `Model judge: ${label(review.verdict || review.status)}. ${review.reason || ''}`));
    const evidence = array(claim.evidence).length ? claim.evidence : array(claim.evidence_ids).map((source_id) => ({source_id}));
    card.append(evidenceLinks(evidence, runId)); return card;
  }
  function proposalCard(proposal, runId, model = false) {
    const card = node('article', 'research-proposal');
    card.append(node('div', 'research-badges', model ? 'MODEL PROPOSAL · UNVERIFIED · NOT LAUNCHED' : 'RECORDED PROPOSAL · NOT LAUNCHED'), node('h4', '', !model && proposal.request === null ? 'Diagnosis before another run' : proposal.title || 'Bounded next experiment'));
    if (typeof proposal.text === 'string') card.append(node('p', 'research-proposal-reason', proposal.text));
    const description = node('dl');
    for (const [key, title] of [['request', 'Proposed configuration'], ['hypothesis', 'Hypothesis'], ['changes', 'What changes'], ['controls', 'Controls'], ['metrics', 'Measures'], ['budget', 'Budget'], ['budget_scope', 'How the budget is counted'], ['seed_selection_scope', 'Seed selection scope'], ['stop_rule', 'Stopping rule'], ['limitations', 'Limitations']]) {
      if (proposal[key] !== undefined && proposal[key] !== null) description.append(node('dt', '', title), node('dd', '', display(proposal[key])));
    }
    card.append(description, evidenceLinks(proposal.evidence || array(proposal.evidence_ids).map((source_id) => ({source_id})), runId));
    if (proposal.provenance) card.append(node('p', 'research-caption', `Provenance: ${display(proposal.provenance)}`));
    return card;
  }
  function renderReport(run) {
    const signature = JSON.stringify({id: run.id, mode: run.mode, question: run.question, report: run.report, sources: run.sources});
    if (signature === reportSignature) return;
    reportSignature = signature;
    const report = run.report;
    byId('results').hidden = !report;
    if (!report) return;
    const publication = publicationFor(report);
    byId('result-mode').textContent = run.mode === 'live' ? 'Live · Token Factory + Tavily' : 'Local inspection · no LLM / no web';
    byId('run-question').textContent = run.question || '';
    byId('answer').textContent = publication
      ? publication.answer || 'No source-backed statement was produced.'
      : 'No source-backed publication is available for this report. Its historical model draft is retained below as unverified material.';
    byId('claims').replaceChildren(...(publication?.facts.length
      ? publication.facts.map((fact) => publicationCard(fact, run.id))
      : [node('p', 'muted', 'No recorded measurements were published.')]));
    byId('literature').replaceChildren(...(publication?.literature_excerpts.length
      ? publication.literature_excerpts.map((fact) => publicationCard(fact, run.id, true))
      : [node('p', 'muted', 'No exact literature excerpts were published.')]));
    const sources = array(run.sources);
    byId('source-count').textContent = `(${sources.length})`;
    byId('sources').replaceChildren(...sources.map((source) => {
      const row = node('div', 'research-source-row'); const description = node('div', '', source.title || source.id);
      description.append(node('small', '', `${label(source.kind)} · ${source.id}`));
      if (source.kind === 'experiment' && source.metadata?.adapter === 'federation_dossier') description.append(node('span', 'research-telemetry-badge', 'Saved coordinator telemetry'));
      row.append(description, citationButton({source_id: source.id}, run.id, 'Inspect snapshot ↗')); return row;
    }));
    const recommendation = publication?.recommendation;
    byId('proposals').replaceChildren(recommendation?.status === 'proposal_only'
      ? proposalCard(recommendation, run.id)
      : node('p', 'muted', 'No bounded next experiment proposal was published.'));
    const limits = publication ? publication.limitations : ['This historical report has not been published through the source-backed reporting protocol. Model approval alone does not establish its claims.'];
    byId('limitations').replaceChildren(...limits.map((item) => node('li', '', display(item))));
    byId('stop-reason').textContent = publication
      ? `Publication: ${publication.status === 'grounded' ? 'source-backed evidence packet' : 'insufficient evidence'}. Scientific efficacy is not established. Automatic policy changes are disabled.`
      : 'Publication: unavailable. Automatic policy changes are disabled.';
    const draft = report.model_draft && typeof report.model_draft === 'object' ? report.model_draft : publication ? {} : report;
    const interpretation = report.model_interpretation || {};
    byId('model-draft').open = false;
    byId('model-answer').textContent = draft.answer || 'No model narrative was retained.';
    renderJudgment({...draft, judge: interpretation.judge || draft.judge}, run.mode);
    const modelClaims = [...array(draft.claims), ...array(interpretation.hypotheses), ...array(interpretation.rejected_hypotheses)];
    byId('model-claims').replaceChildren(...modelClaims.map((claim) => modelClaimCard(typeof claim === 'string' ? {text: claim} : claim, run.id)));
    byId('model-proposals').replaceChildren(...array(draft.next_runs).map((proposal) => proposalCard(proposal, run.id, true)));
    const gaps = node('ul');
    gaps.append(...[...array(interpretation.gaps), ...array(draft.limitations)].map((gap) => node('li', '', display(gap))));
    byId('model-gaps').replaceChildren(node('h4', '', 'Model-reported gaps and limitations'), gaps);
    if (draft.stop_reason) byId('model-gaps').append(node('p', '', `Model stopping reason: ${display(draft.stop_reason)}`));
  }
  function render(data) {
    snapshot = data;
    const configured = data.configured || {};
    for (const [name, title] of [['nebius', 'Token Factory'], ['tavily', 'Tavily']]) {
      byId(name).textContent = `${title} · ${configured[name] ? 'configured' : 'key needed'}`;
      byId(name).className = configured[name] ? 'configured' : 'missing';
    }
    byId('provider').textContent = `Model: ${configured.model || 'not configured'} · Credentials stay on the server.${configured.base_url ? ` Endpoint: ${configured.base_url}` : ''}`;
    renderCatalog(array(data.catalog));
    const run = data.run || {};
    byId('status').textContent = label(run.status || 'ready');
    byId('mode').textContent = data.active ? 'Research running' : 'Ready';
    byId('run-meta').textContent = run.id ? `${run.mode === 'live' ? 'Live research' : 'Local inspection'} · Run ${run.id}` : 'No research run yet.';
    byId('run-context').hidden = !run.id;
    byId('saved-question').textContent = run.id ? run.question || 'Question not recorded for this saved run.' : '';
    byId('saved-shadow-permission').textContent = !run.id ? '' : run.mode === 'offline'
      ? 'Shadow permission: unavailable in local inspection; no training runs.'
      : run.allow_shadow === true
        ? 'Shadow permission: enabled for this run, for at most one bounded pilot. Permission does not mean a pilot ran.'
        : run.allow_shadow === false
          ? 'Shadow permission: disabled for this run; no pilot authorized.'
          : 'Shadow permission: not recorded for this saved run.';
    byId('export').hidden = !run.id;
    if (run.id) byId('export').href = endpoint('export', {run_id: run.id});
    renderTrace(array(run.events), run); renderReport(run); controls();
    showError(run.error || data.error || data.catalog_error || '');
    if (data.active && (run.allow_shadow || byId('allow-shadow').checked)) watchShadow?.();
  }
  function schedule() {
    clearTimeout(timer); timer = null;
    if (snapshot?.active) timer = setTimeout(refresh, 2000);
  }
  async function refresh() {
    if (requesting) return;
    requesting = true; byId('refresh').disabled = true; controls();
    try { render(await request('/api/research')); }
    catch (error) { showError(error.message); byId('status').textContent = 'Connection interrupted'; }
    finally { requesting = false; byId('refresh').disabled = false; controls(); schedule(); }
  }
  async function submit(mode) {
    if (submitting || requesting || snapshot?.active || !byId('form').reportValidity()) return;
    if (mode === 'live' && (!snapshot?.configured?.nebius || !snapshot?.configured?.tavily)) return;
    if (!validSelection()) return;
    submitting = true; controls(); showError('');
    try {
      const result = await request('/api/research', {question: byId('question').value.trim(), record_ids: [...selected], mode, allow_shadow: mode === 'live' && byId('allow-shadow').checked === true});
      if (result.configured && result.catalog) render(result);
      else await refresh();
    } catch (error) { showError(error.message); }
    finally { submitting = false; controls(); schedule(); }
  }
  async function openSource(runId, evidence) {
    const serial = ++sourceRequest;
    const dialog = byId('source-dialog');
    byId('source-title').textContent = 'Loading saved source…';
    byId('source-body').replaceChildren(node('p', 'muted', 'Reading the immutable evidence snapshot.'));
    if (!dialog.open) dialog.showModal();
    try {
      const source = await request(endpoint('source', {run_id: runId, source_id: evidence.source_id}));
      if (serial !== sourceRequest) return;
      byId('source-title').textContent = source.title || source.id;
      const content = node('div');
      content.append(node('p', 'research-source-digest', `${label(source.kind)} · ${source.id}`));
      if (source.kind === 'experiment' && source.metadata?.adapter === 'federation_dossier') content.append(node('p', 'research-telemetry-badge', 'Saved coordinator telemetry · not a live control connection'));
      if (evidence.quote) {
        content.append(node('h4', '', 'Cited source text or value'), node('blockquote', '', evidence.quote));
        if (typeof evidence.locator === 'string' && evidence.locator.startsWith('/')) {
          content.append(node('p', 'research-source-digest', 'This citation identifies a JSON value at the recorded pointer below. The full saved record is shown for inspection; matching text alone would not verify the claim.'));
        } else {
          const matches = String(source.content || '').includes(evidence.quote);
          content.append(node('p', 'research-source-digest', matches ? 'Exact excerpt found in this saved source. This checks text provenance, not whether the claim follows from it.' : 'Exact excerpt was not found in this snapshot. Review the claim and source before relying on it.'));
        }
      }
      if (evidence.locator) content.append(node('p', '', `Locator: ${display(evidence.locator)}`));
      const url = webURL(source.uri);
      if (url) { const link = node('a', '', 'Open original source ↗'); link.href = url; link.target = '_blank'; link.rel = 'noopener noreferrer'; content.append(link); }
      else if (source.uri) content.append(node('p', 'research-source-digest', source.uri));
      content.append(node('p', 'research-source-digest', `SHA-256: ${source.sha256 || 'not supplied'}`));
      const meta = node('details'); meta.append(node('summary', '', 'Source provenance'), node('pre', '', display(source.metadata || {}))); content.append(meta);
      content.append(node('h4', '', 'Saved source content'), node('pre', '', display(source.content)));
      byId('source-body').replaceChildren(content);
    } catch (error) { if (serial === sourceRequest) { byId('source-title').textContent = 'Source unavailable'; byId('source-body').replaceChildren(node('p', 'error', error.message)); } }
  }
  function initShadow() {
    const field = (id) => byId(`shadow-${id}`);
    if (!field('form')) return;
    let state = null;
    let busy = false;
    let fetching = false;
    let dirty = false;
    let poll = null;
    let knownId = null;
    let catalogKey = '';
    let eventKey = '';
    const changes = () => array(state?.catalog?.changes);
    const choice = () => changes().find((item) => item.id === field('change').value);
    const error = (message) => { field('error').textContent = message || ''; field('error').hidden = !message; };
    function manageControls() {
      const active = Boolean(state?.active);
      const available = state?.runtime?.available === true;
      const working = busy || fetching;
      field('prepare').disabled = active || working || !available || !choice();
      field('start').disabled = active || working || !available || dirty || state?.run?.status !== 'frozen' || !state?.run?.id;
      for (const name of ['change', 'seeds', 'rounds']) field(name).disabled = active || working || !changes().length;
      field('value').disabled = active || working || field('value').hidden;
      field('value-option').disabled = active || working || field('value-option').hidden;
      field('refresh').disabled = working;
      field('prepare').textContent = busy ? 'Saving trial request…' : 'Freeze trial configuration →';
      field('start').textContent = active ? 'Local trial running…' : 'Run frozen trial ↗';
      if (dirty && state?.run?.status === 'frozen') field('state-note').textContent = 'Settings changed. Freeze a new trial before running; the manifest below still belongs to the earlier frozen trial.';
    }
    function configureValue(value) {
      const spec = choice();
      if (!spec) return;
      const options = array(spec.options);
      const isChoice = options.length > 0;
      field('value').hidden = isChoice;
      field('value-option').hidden = !isChoice;
      if (isChoice) {
        field('value-option').replaceChildren(...options.map((item) => {
          const option = node('option', '', typeof item === 'object' ? item.label || item.value : item);
          option.value = String(typeof item === 'object' ? item.value : item); return option;
        }));
        field('value-option').value = String(value ?? spec.default ?? options[0]);
      } else {
        field('value').min = spec.min === undefined ? '' : String(spec.min);
        field('value').max = spec.max === undefined ? '' : String(spec.max);
        field('value').value = String(value ?? spec.default ?? (spec.id === 'td3_actor_lr' ? 0.00005 : spec.baseline ?? 0));
      }
      field('baseline').textContent = `Baseline ${spec.label || label(spec.id)}: ${display(spec.baseline ?? 'recorded in manifest')}. Baseline TD3 critic inputs: normalized.`;
      manageControls();
    }
    function renderCatalog() {
      const key = JSON.stringify(changes());
      if (key === catalogKey) return;
      catalogKey = key;
      const previous = field('change').value;
      field('change').replaceChildren(...changes().map((item) => { const option = node('option', '', item.label || label(item.id)); option.value = item.id; return option; }));
      field('change').value = changes().some((item) => item.id === previous) ? previous : changes().some((item) => item.id === 'td3_actor_lr') ? 'td3_actor_lr' : changes()[0]?.id || '';
      configureValue();
    }
    function rows(target, heading, values) {
      target.replaceChildren();
      if (values == null) return;
      target.append(node('h4', '', heading));
      if (typeof values !== 'object' || Array.isArray(values)) { target.append(node('p', '', display(values))); return; }
      const list = node('dl');
      for (const [key, value] of Object.entries(values)) list.append(node('dt', '', label(key)), node('dd', '', display(value)));
      target.append(list);
    }
    function renderComparison(comparison) {
      field('results').hidden = !comparison;
      if (!comparison) return;
      const cards = [];
      if (typeof comparison === 'object' && !Array.isArray(comparison)) {
        const branches = comparison.branches || {};
        const arms = ['local', 'reset_only', 'uniform', 'clipped'];
        const names = {local: 'Local', reset_only: 'Reset-only', uniform: 'Uniform federation', clipped: 'Clipped federation'};
        const seeds = [...new Set([...array(comparison.expected_training_seeds).map(String), ...Object.keys(branches.baseline?.seeds || {}), ...Object.keys(branches.candidate?.seeds || {})])];
        const number = (value, digits = 5) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'Unavailable';
        const status = (seed, raw) => !seed ? 'Missing seed' : !raw ? `${label(seed.status || 'missing')} · missing arm` : raw.valid_complete === true ? `Valid endpoint · competence ${raw.competent === true ? 'reached' : raw.competent === false ? 'not reached' : 'unreported'}` : 'Invalid / incomplete endpoint';
        const card = (title, note) => {
          const result = node('article', 'research-shadow-result-card research-shadow-wide');
          result.append(node('h4', '', title));
          if (note) result.append(node('p', 'research-caption', note));
          cards.push(result); return result;
        };
        const table = (parent, caption, headings) => {
          const wrap = node('div', 'research-shadow-table-wrap');
          const result = node('table', 'research-shadow-table');
          result.append(node('caption', '', caption));
          const head = node('thead'), row = node('tr'), body = node('tbody');
          for (const heading of headings) { const cell = node('th', '', heading); cell.scope = 'col'; row.append(cell); }
          head.append(row); result.append(head, body); wrap.append(result); parent.append(wrap); return body;
        };
        const addRow = (body, values, className = '') => {
          const row = node('tr', className);
          for (const value of values) row.append(node('td', '', value));
          body.append(row);
        };
        if (seeds.length) {
          const rawCard = card('Raw learner endpoints', 'Each row compares the same control and training seed under the baseline and candidate setting. Lower cost and errors are better. A positive cost change means the candidate was worse; these are endpoints, not shots-to-competence.');
          const rawTable = table(rawCard, 'Baseline → candidate, before any gate fallback', ['Seed', 'Control', 'Baseline cost', 'Candidate cost', 'Cost change', 'Profile RMSE (keV)\nBaseline → candidate', 'βN MAE\nBaseline → candidate', 'Recorded validity']);
          for (const seed of seeds) {
            const baseline = branches.baseline?.seeds?.[seed], candidate = branches.candidate?.seeds?.[seed];
            for (const arm of arms) {
              const before = baseline?.arms?.[arm]?.raw, after = candidate?.arms?.[arm]?.raw;
              const pair = array(comparison.candidate_vs_baseline_raw_endpoint?.[arm]?.by_seed).find((item) => String(item.seed) === seed);
              const change = before?.valid_complete === true && after?.valid_complete === true && baseline?.status === 'complete' && candidate?.status === 'complete' && pair?.comparable === true && typeof pair.relative_cost_change === 'number' && Number.isFinite(pair.relative_cost_change) ? `${pair.relative_cost_change > 0 ? '+' : ''}${(pair.relative_cost_change * 100).toFixed(2)}%` : 'Not comparable';
              addRow(rawTable, [seed, names[arm], number(before?.mean_cost), number(after?.mean_cost), change,
                `${number(before?.mean_Te_profile_rmse_keV)} → ${number(after?.mean_Te_profile_rmse_keV)}`,
                `${number(before?.mean_beta_mae, 6)} → ${number(after?.mean_beta_mae, 6)}`,
                `Baseline: ${status(baseline, before)}; candidate: ${status(candidate, after)}`], before?.valid_complete === true && after?.valid_complete === true ? '' : 'research-shadow-row-incomplete');
            }
          }
          const gateCard = card('Gate choices, kept separate', 'A passing gate selects the raw proposal for this diagnostic report. A rejected gate selects the incumbent fallback. This is not a separately trained deployed branch or an activated controller.');
          const gateTable = table(gateCard, 'Recorded federation gate and selected endpoint', ['Configuration', 'Seed', 'Control', 'Gate decision', 'Raw cost', 'Gate-selected cost', 'Raw competence']);
          for (const variant of ['baseline', 'candidate']) for (const seed of seeds) for (const arm of ['uniform', 'clipped']) {
            const record = branches[variant]?.seeds?.[seed]?.arms?.[arm];
            const gate = record?.gate;
            const decision = gate?.accept === true ? 'Accepted · raw selected' : gate?.accept === false ? 'Rejected · incumbent fallback' : 'Not recorded';
            const reason = gate?.reason ? ` (${display(gate.reason)})` : '';
            addRow(gateTable, [label(variant), seed, names[arm], decision + reason,
              number(record?.raw?.mean_cost), number(record?.deployed?.mean_cost),
              record?.raw?.competent === true ? 'Reached' : record?.raw?.competent === false ? 'Not reached' : 'Unavailable'], !record || gate?.accept === false ? 'research-shadow-row-incomplete' : '');
          }
          const accounting = card('Completed work & failures', 'Training includes donor and target episodes. Gate and reporting episodes count as evaluation. Optimizer-only work remains in the full recorded comparison.');
          const counts = node('ul', 'research-shadow-summary-list');
          for (const variant of ['baseline', 'candidate']) {
            const branch = branches[variant], work = branch?.accounting || {};
            const count = (value) => Number.isInteger(value) && value >= 0 ? String(value) : 'unavailable';
            counts.append(node('li', '', `${label(variant)}: ${label(branch?.status || 'missing')}; ${count(work.training_episodes)} training and ${count(work.evaluation_episodes)} evaluation episodes recorded.`));
            for (const seed of seeds) {
              const record = branch?.seeds?.[seed];
              counts.append(node('li', '', `${label(variant)}, seed ${seed}: ${label(record?.status || 'missing')}; training stability ${record?.training_stable === true ? 'passed recorded checks' : record?.training_stable === false ? 'failed recorded checks' : 'not recorded'}.`));
            }
          }
          const failures = array(comparison.failures);
          counts.append(node('li', '', failures.length ? `${failures.length} recorded failure entries:` : 'No failure entries recorded. Missing or incomplete rows above still prevent a complete comparison.'));
          failures.forEach((failure) => counts.append(node('li', 'research-shadow-failure', display(failure))));
          accounting.append(counts);
        } else {
          card('Paired endpoint data unavailable', 'The saved record does not contain per-seed baseline and candidate endpoints. Inspect the full record below; no improvement has been inferred.');
        }
        const limits = card('What this result can support');
        const list = node('ul', 'research-shadow-summary-list');
        const limitations = array(comparison.limitations);
        (limitations.length ? limitations : ['No limitations were supplied in this record. Treat this small development pilot as preliminary.']).forEach((item) => list.append(node('li', '', display(item))));
        array(comparison.fairness?.reasons).forEach((reason) => list.append(node('li', 'research-shadow-failure', `Comparison issue: ${display(reason)}`)));
        limits.append(list);
        if (comparison.provenance?.trial_id) limits.append(node('p', 'research-source-digest', `Saved trial: ${comparison.provenance.trial_id}`));
        if (comparison.provenance?.result_sha256) limits.append(node('p', 'research-source-digest', `Worker result SHA-256: ${comparison.provenance.result_sha256}`));
        const recommendation = comparison.recommendation;
        field('recommendation').textContent = `Pilot recommendation: ${typeof recommendation?.text === 'string' ? recommendation.text : typeof recommendation === 'string' ? recommendation : 'No recommendation recorded.'}`;
        const proposal = comparison.next_run_proposal;
        if (proposal && typeof proposal === 'object' && !Array.isArray(proposal)) {
          const next = card('Proposed next run · not launched', 'This is a saved proposal. It has not started a trial or changed a controller.');
          if (proposal.request && typeof proposal.request === 'object') {
            const request = proposal.request, budget = proposal.budget || {};
            const settings = node('dl', 'research-shadow-proposal-fields');
            const change = changes().find((item) => item.id === request.change);
            for (const [title, value] of [
              ['Candidate change', `${change?.label || label(request.change)} = ${display(request.value)}`],
              ['Paired training seeds', array(request.seeds).join(', ') || 'Not supplied'],
              ['Rounds per arm', request.rounds ?? 'Not supplied'],
              ['Planned episodes, both configurations', budget.total_episodes ?? 'Not supplied'],
              ['Maximum wall time', budget.maximum_wall_seconds === undefined ? 'Not supplied' : `${budget.maximum_wall_seconds} seconds`]
            ]) settings.append(node('dt', '', title), node('dd', '', value));
            next.append(settings);
          } else next.append(node('p', 'research-shadow-failure', 'No runnable configuration was proposed. Resolve the diagnostic gap before preparing another trial.'));
          if (proposal.reason) next.append(node('p', '', display(proposal.reason)));
          if (proposal.seed_selection_scope) next.append(node('p', 'research-caption', display(proposal.seed_selection_scope)));
        }
        const complete = comparison.status === 'complete' && !['failed', 'timed_out', 'interrupted'].includes(comparison.execution_status);
        if (!complete) {
          const incomplete = node('p', 'research-shadow-incomplete research-shadow-wide', `Comparison status: ${label(comparison.status || 'not recorded')}. Retain partial results and failures; this record cannot establish an improvement.`);
          cards.unshift(incomplete);
        }
      } else {
        field('recommendation').textContent = 'Inspect the recorded comparison before choosing a follow-up.';
      }
      const full = node('details', 'research-shadow-full-record research-shadow-wide');
      full.append(node('summary', '', 'Full recorded comparison · JSON'), node('pre', '', display(comparison))); cards.push(full);
      field('comparison').replaceChildren(...cards);
    }
    function renderState(data) {
      state = data;
      renderCatalog();
      const run = data.run || {};
      const status = run.status || 'idle';
      field('status').textContent = label(status);
      field('runtime').textContent = data.runtime?.available === true
        ? 'Local RL Python environment found. The frozen worker validates simulator startup. No Nebius or Tavily key is used for the pilot.'
        : `Local runtime unavailable: ${data.runtime?.reason || 'runtime availability has not been reported'}`;
      field('id').textContent = run.id ? `Trial ${run.id}` : 'No frozen trial';
      if (run.id && run.id !== knownId) {
        knownId = run.id;
        if (!dirty && run.config) {
          if (changes().some((item) => item.id === run.config.change)) field('change').value = run.config.change;
          configureValue(run.config.value);
          field('seeds').value = array(run.config.seeds).join(', ') || '17';
          field('rounds').value = String(run.config.rounds ?? 1);
        }
      }
      const frozen = run.config ? {
        change: label(run.config.change),
        baseline: run.manifest?.baseline?.[run.config.change] ?? 'See frozen manifest',
        candidate: run.config.value,
        heating_critic_coordinates: run.manifest?.baseline?.heating_critic_coordinates ?? 'See frozen manifest',
        paired_training_seeds: run.config.seeds,
        rounds_per_arm: run.config.rounds
      } : run.id ? {} : null;
      rows(field('config'), 'Frozen baseline → candidate', frozen);
      if (!run.id) field('config').append(node('p', 'muted', 'The frozen baseline and candidate will appear here.'));
      rows(field('budget'), 'Frozen pilot budget', run.budget);
      field('provenance').hidden = !run.manifest;
      field('manifest').textContent = display(run.manifest || {});
      field('export').hidden = !run.id;
      if (run.id) field('export').href = `/api/shadow/export?${new URLSearchParams({id: run.id})}`;
      field('stage-frozen').className = run.manifest || status === 'frozen' ? 'observed' : '';
      field('stage-running').className = data.active || ['running', 'completed', 'timed_out', 'interrupted'].includes(status) ? 'observed' : '';
      field('stage-completed').className = run.comparison ? 'observed' : '';
      const notes = {
        frozen: 'Source and configuration are frozen. Run uses the exact manifest shown here.',
        running: 'Running the isolated baseline and candidate. Partial results are not a completed comparison.',
        completed: 'The local pilot has finished. Review paired outcomes, costs and failure accounting below.',
        failed: 'This trial failed. Preserve its source and failure record when deciding the next test.',
        timed_out: 'The pilot reached its time limit. Unfinished comparisons cannot establish an improvement.',
        interrupted: 'The trial was interrupted. Its recorded state is retained.'
      };
      field('state-note').textContent = notes[status] || 'Preparing a trial records its inputs and budget. It does not run physics.';
      const progress = run.progress;
      const hasProgress = progress && Number.isInteger(progress.completed_episodes) && progress.completed_episodes >= 0 && Number.isInteger(progress.maximum_episodes) && progress.maximum_episodes > 0;
      field('progress').hidden = !hasProgress;
      if (hasProgress) {
        field('progress-label').textContent = `${progress.completed_episodes} / ${progress.maximum_episodes} episodes completed${progress.phase ? ` · ${display(progress.phase)}` : ''}. ${status === 'running' ? 'The comparison is still in progress.' : 'Recorded worker progress.'}`;
        field('progress-bar').max = progress.maximum_episodes;
        field('progress-bar').value = Math.min(progress.completed_episodes, progress.maximum_episodes);
      }
      const events = array(run.events);
      const key = JSON.stringify(events);
      if (key !== eventKey) {
        eventKey = key;
        field('events').replaceChildren(...events.map((event, index) => {
          const item = node('article', 'research-shadow-event');
          item.append(node('strong', '', `${String(index + 1).padStart(2, '0')} · ${label(event.kind || event.stage || event.type || 'event')}`), node('p', '', display(event.message || event.detail || event)));
          const details = node('details'); details.append(node('summary', '', 'Recorded event'), node('pre', '', display(event))); item.append(details); return item;
        }));
      }
      renderComparison(run.comparison);
      error(run.error || data.error || '');
      pilotHint();
      manageControls();
    }
    function scheduleShadow() {
      clearTimeout(poll);
      poll = state?.active || (snapshot?.active && snapshot?.run?.allow_shadow) ? setTimeout(refreshShadow, 2500) : null;
    }
    async function refreshShadow() {
      if (fetching) return;
      fetching = true; manageControls();
      try { renderState(await request('/api/shadow')); }
      catch (failure) { error(failure.message); field('status').textContent = 'Connection interrupted'; }
      finally { fetching = false; manageControls(); scheduleShadow(); }
    }
    function selectedConfig() {
      const tokens = field('seeds').value.trim().split(/[,\s]+/);
      const seeds = tokens.map((token) => /^\d+$/.test(token) ? Number(token) : NaN);
      if (seeds.length < 1 || seeds.length > 3 || new Set(seeds).size !== seeds.length || seeds.some((seed) => !Number.isSafeInteger(seed) || seed < 0 || seed > 1000000)) throw new Error('Choose 1–3 distinct integer seeds from 0–1,000,000, separated by commas.');
      const rounds = Number(field('rounds').value);
      if (![1, 2].includes(rounds)) throw new Error('Choose one or two rounds for this pilot.');
      const spec = choice();
      if (!spec) throw new Error('Choose a supported candidate field.');
      const value = array(spec.options).length ? field('value-option').value : Number(field('value').value);
      if (!array(spec.options).length && (!field('value').value.trim() || !Number.isFinite(value) || (spec.min !== undefined && value < spec.min) || (spec.max !== undefined && value > spec.max))) throw new Error('The candidate value must be within the displayed field limits.');
      if (value === spec.baseline) throw new Error('Choose a candidate value different from the baseline.');
      return {change: spec.id, value, seeds, rounds};
    }
    async function freezeShadow() {
      if (busy || fetching || state?.active || !state?.runtime?.available || !field('form').reportValidity()) return;
      let config;
      try { config = selectedConfig(); } catch (failure) { error(failure.message); return; }
      busy = true; error(''); manageControls();
      try { const result = await request('/api/shadow/prepare', config); dirty = false; renderState(result); }
      catch (failure) { error(failure.message); }
      finally { busy = false; manageControls(); scheduleShadow(); }
    }
    async function startShadow() {
      if (busy || fetching || state?.active || dirty || !state?.runtime?.available || state?.run?.status !== 'frozen' || !state?.run?.id) return;
      busy = true; error(''); manageControls();
      try { renderState(await request('/api/shadow/start', {id: state.run.id})); }
      catch (failure) { error(failure.message); }
      finally { busy = false; manageControls(); scheduleShadow(); }
    }
    function pilotHint() {
      const tokens = field('seeds').value.trim().split(/[,\s]+/);
      const count = tokens.length;
      const rounds = Number(field('rounds').value);
      if (count >= 1 && count <= 3 && new Set(tokens.map(Number)).size === count && [1, 2].includes(rounds) && tokens.every((token) => /^\d+$/.test(token) && Number(token) <= 1000000)) {
        field('budget-hint').textContent = `Planned for both configurations: ${count * (20 + 36 * rounds)} TORAX episodes (${count * (16 + 16 * rounds)} training + ${count * (4 + 20 * rounds)} evaluation). Torch/BLAS thread limit: 1; JAX may use additional threads. Shared CPU, 600-second limit. Seeds must be distinct integers from 0–1,000,000. Counts are planned, not completed.`;
      } else field('budget-hint').textContent = 'Choose 1–3 distinct integer seeds from 0–1,000,000 and 1–2 rounds. Torch/BLAS thread limit: 1; JAX may use additional threads. Shared CPU, 600-second limit.';
    }
    function changed() { dirty = true; pilotHint(); manageControls(); }
    field('change').addEventListener('change', () => { configureValue(); changed(); });
    for (const id of ['value', 'value-option', 'seeds', 'rounds']) field(id).addEventListener('input', changed);
    field('form').addEventListener('submit', (event) => { event.preventDefault(); freezeShadow(); });
    field('start').addEventListener('click', startShadow);
    field('refresh').addEventListener('click', refreshShadow);
    watchShadow = refreshShadow;
    refreshShadow();
  }
  byId('form').addEventListener('submit', (event) => { event.preventDefault(); submit('live'); });
  byId('offline').addEventListener('click', () => submit('offline'));
  byId('refresh').addEventListener('click', refresh);
  byId('source-close').addEventListener('click', () => byId('source-dialog').close());
  document.querySelectorAll('[data-research-example]').forEach((button) => button.addEventListener('click', () => { byId('question').value = examples[button.dataset.researchExample]; byId('question').focus(); }));
  // Only a read on load. Research and model calls always require explicit submission.
  initShadow();
  refresh();
})();
