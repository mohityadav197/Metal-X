from __future__ import annotations

import time

import streamlit as st

from ui.platform import (
	BACKEND_OFFLINE_MESSAGE,
	REQUEST_TIMEOUT_MESSAGE,
	configure_page,
	get_json,
	inject_styles,
	post_json,
	render_sidebar,
)


configure_page("Home | METALLURGIC-X")
inject_styles()
render_sidebar("Executive Summary")


def _init_home_session_state() -> None:
	defaults = {
		"pulse_ok": False,
		"pulse_err": None,
		"home_last_backend_payload": {},
		"document_embedded": False,
	}
	for key, value in defaults.items():
		if key not in st.session_state:
			st.session_state[key] = value


_init_home_session_state()


@st.cache_data(ttl=3, show_spinner=False)
def _cached_system_payload(timeout: float) -> tuple[bool, dict, str | None]:
	ok, data, err = get_json("/system/intelligence", timeout=timeout)
	return ok, (data or {}), err


def system_monitor(timeout: float = 2.0) -> tuple[bool, dict, str | None, float]:
	started = time.perf_counter()
	try:
		ok, data, err = _cached_system_payload(timeout)
		st.session_state["home_last_backend_payload"] = data
		elapsed = time.perf_counter() - started
		return ok, (data or {}), err, elapsed
	except Exception:
		elapsed = time.perf_counter() - started
		return False, {}, BACKEND_OFFLINE_MESSAGE, elapsed


st.markdown("<div class='hero-wrap'><h1 class='brand-title'>METALLURGIC-X</h1></div>", unsafe_allow_html=True)
st.markdown(
	"<p class='hero-subtext'>Traditional alloy design takes months; Metallurgic-X synthesizes aerospace-grade alloys in seconds.</p>",
	unsafe_allow_html=True,
)


def render_live_pulse() -> tuple[bool, str | None]:
	pulse_host = st.container()
	warmup_host = pulse_host.empty()
	ok_pulse, _, pulse_err, elapsed = system_monitor(timeout=5.0)

	if elapsed > 1.5:
		warmup_host.info("Engines Warming Up... (Backend is currently loading Neural Models)")
	else:
		warmup_host.empty()

	pulse_class = "online" if ok_pulse else "offline"
	pulse_label = "Live Pulse Online" if ok_pulse else "Live Pulse Offline"
	pulse_host.markdown(
		(
			"<div class='pulse-indicator'>"
			f"<span class='pulse-dot {pulse_class}'></span>{pulse_label}"
			"</div>"
		),
		unsafe_allow_html=True,
	)
	return ok_pulse, pulse_err


if hasattr(st, "fragment"):

	@st.fragment
	def _render_pulse_fragment() -> None:
		ok, err = render_live_pulse()
		st.session_state["pulse_ok"] = ok
		st.session_state["pulse_err"] = err


	_render_pulse_fragment()
else:
	ok, err = render_live_pulse()
	st.session_state["pulse_ok"] = ok
	st.session_state["pulse_err"] = err

st.markdown("### Impact Dashboard")
metric_a, metric_b, metric_c = st.columns(3)
with metric_a:
	st.markdown(
		"""
<div class='glass-card'>
	<div class='impact-label'>Alloy Candidates</div>
	<div class='impact-value blue'>500+ Alloys</div>
</div>
""",
		unsafe_allow_html=True,
	)
with metric_b:
	st.markdown(
		"""
<div class='glass-card'>
	<div class='impact-label'>Synthesis Speed</div>
	<div class='impact-value cyan'>0.4 sec</div>
</div>
""",
		unsafe_allow_html=True,
	)
with metric_c:
	st.markdown(
		"""
<div class='glass-card'>
	<div class='impact-label'>Cycle Reduction</div>
	<div class='impact-value green'>35% Faster</div>
</div>
""",
		unsafe_allow_html=True,
	)

st.markdown(
	"""
<div class='glass-card'>
	<h3>Problem Statement</h3>
	<p>
		Traditional alloy design takes months; Metallurgic-X synthesizes aerospace-grade alloys in seconds.
	</p>
</div>
""",
	unsafe_allow_html=True,
)

st.markdown("### Data Intake")

st.sidebar.markdown("### Data Intake")
with st.sidebar.container():
	uploaded_files = st.file_uploader(
		"Upload Alloy Specs & Papers (PDF)", 
		type=["pdf"], 
		accept_multiple_files=True
	)
	if st.sidebar.button("Process Research Paper"):
		if not uploaded_files:
			st.sidebar.warning("Please select at least one PDF.")
		else:
			import requests
			# Format file payload matching FastAPI expecting List[UploadFile] named 'files'
			files_payload = [
				("files", (file.name, file.getvalue(), "application/pdf")) 
				for file in uploaded_files
			]
			with st.sidebar.spinner("Embedding document into ChromaDB..."):
				try:
					response = requests.post(
						"http://127.0.0.1:8000/upload", 
						files=files_payload,
						timeout=60
					)
					if response.status_code == 200:
						st.sidebar.success("Indexing successful!")
						st.sidebar.json(response.json())
						st.session_state.document_embedded = True
					else:
						st.sidebar.error(f"Error {response.status_code}: {response.text}")
				except Exception as e:
					st.sidebar.error(f"Transmission failed: {str(e)}")

# Check document state for the main chat UI area
if not st.session_state.document_embedded:
	st.warning("Please upload and process a research document in the sidebar to begin.")
else:
	st.success("Document embedded! You can now issue queries to the Research Endpoint.")
	query = st.chat_input("Ask the intelligence endpoint...")
	if query:
		with st.spinner("Querying specific knowledge base..."):
			ok, data, err = post_json(
				"/research",
				{"query": query, "top_k": 5},
				timeout=10.0,
			)
			if ok:
				st.markdown(f"**Query**: {query}")
				st.write(data.get("answer", data.get("response", "No answer provided.")))
				if "citations" in data:
					st.caption(f"Citations: {', '.join(data['citations'])}")
			else:
				st.error(f"Failed to fetch. Reason: {err}")

st.markdown("### Quick Start")
col_left, col_right = st.columns([1.1, 1.0])
with col_left:
	if st.button("Run AI Synthesis"):
		ok, data, err = post_json("/synthesize", {"target_strength": 320.0}, timeout=5.0)
		if ok:
			st.success("Success! New alloy recipe created.")
			st.metric("Target Strength", f"{data.get('target_strength', 0):.1f} MPa")
			st.metric("Features Returned", len(data.get("feature_order", [])))
		else:
			if err == REQUEST_TIMEOUT_MESSAGE:
				st.warning("Synthesis request exceeded 5s timeout. Please retry.")
			else:
				st.error(BACKEND_OFFLINE_MESSAGE if err == BACKEND_OFFLINE_MESSAGE else f"Request failed: {err}")

	if st.button("Refresh Live Pulse"):
		_cached_system_payload.clear()
		ok, data, err = get_json("/system/intelligence", timeout=5.0)
		if ok:
			st.success("Backend heartbeat received.")
			st.session_state["home_last_backend_payload"] = data
			st.metric("CPU Usage", f"{float(data.get('cpu_usage', 0.0)):.1f}%")
		else:
			if err == REQUEST_TIMEOUT_MESSAGE:
				st.warning("Live pulse request exceeded 5s timeout. Please retry.")
			else:
				st.error(BACKEND_OFFLINE_MESSAGE)

with col_right:
	st.markdown(
		"""
<div class='glass-card'>
	<b>Live Pulse</b><br/>
	Monitoring backend telemetry from /system/intelligence.
</div>
""",
		unsafe_allow_html=True,
	)

pulse_err = st.session_state.get("pulse_err")
if pulse_err == BACKEND_OFFLINE_MESSAGE:
	st.info(BACKEND_OFFLINE_MESSAGE)

st.markdown(
	"""
<div class='glass-card' style='margin-top:0.7rem;'>
	<b>Guide</b>
	<ul class='guide-list'>
		<li>Set your target in Synthesis Lab.</li>
		<li>Review feature outputs and integrity checks.</li>
		<li>Use Research Hub for literature-backed validation.</li>
	</ul>
</div>
""",
	unsafe_allow_html=True,
)

st.sidebar.divider()
st.sidebar.markdown("### 📬 Get in Touch")
st.sidebar.markdown("Queries and feedback are welcome!")
st.sidebar.markdown("[**📧 Click here to email me**](mailto:mohitkhairwal2005@gmail.com)")

with st.expander("⚙️ Core Tech Stack & System Flow"):
	st.markdown(
		"""
- **Frontend**: Streamlit (User interface & multipart data handling)
- **Backend Engine**: FastAPI (Orchestration & API endpoints)
- **Knowledge Base (RAG)**: ChromaDB (800/100 chunking with Semantic Distance Filtering)
- **Generative Synthesis**: Groq API powered by llama-3.1-8b-instant for high-speed, physics-aware reasoning.
		"""
	)
	st.info("Usage Disclaimer: This is a limited-access prototype subject to Groq API rate limits and local compute constraints.")

