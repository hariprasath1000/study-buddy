"""Trip-planning AI agent: Claude + MCP tools (travel_server.py). Car and bike only.

    export ANTHROPIC_API_KEY=...
    python agent.py

Optional: ENABLE_WEB_SEARCH=1 lets Claude look up current entry fees / timings
(needs web search enabled for your Anthropic organisation).
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).parent
MODEL = os.environ.get("TRAVEL_AGENT_MODEL", "claude-sonnet-5-5")
MAX_ROUNDS = 14          # guardrail: tool-use rounds per user message
TOOL_TIMEOUT = 90        # seconds per tool call
WEB_SEARCH = os.environ.get("ENABLE_WEB_SEARCH") == "1"

# stdio_client only forwards a few env vars to the server, so pass proxy / CA settings explicitly.
FORWARD_ENV = ["TRAVEL_CONTACT", "OSRM_URL", "TRAVEL_PREFS_PATH", "SSL_CERT_FILE", "SSL_CERT_DIR",
               "REQUESTS_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"]

VEHICLE_ALIASES = {"car": "car", "bike": "bike", "motorbike": "bike", "motorcycle": "bike", "two-wheeler": "bike"}


def build_system(web_search: bool) -> str:
    fees = ("You also have a web_search tool: use it to confirm entry fees and timings for the places you recommend "
            "(a few searches, not one per place). Prefer official sources." if web_search else
            "You have no web access, so fees and timings come from your own knowledge and must be flagged as approximate.")
    return f"""You are a friendly road-trip planning assistant. You plan CAR and BIKE (motorcycle / two-wheeler) trips ONLY.
If asked about a bus, train, flight, walking or a bicycle, say you only plan car and bike trips.
The user has already told you their vehicle in the first message.

STEP 1 - Get trip details. Call get_preferences first and reuse anything saved (say so briefly).
Required: start, destination, departure date and time. Optional: interests (heritage, nature, temples, museums,
viewpoints, family, food), time available, how far they will detour, mileage and fuel price, number of people,
places they already want to visit. Ask only for what is missing, in one short message. When they give reusable
details (mileage, tank size, fuel price, interests, people), offer to save them with save_preferences.

STEP 2 - Routes. Call plan_routes. Tell the user how many route options exist, then for EACH option give: distance,
time, the main roads in order, the towns it passes (with km markers) and how it differs from the fastest. Recommend
one with a reason and let them choose; they can simply type "1", "the fastest" or a road name.

STEP 3 - Tourist places along the chosen route. YOU find these, from your own knowledge of the towns and districts the
route passes (use the towns list and km markers). Do not ask for a map search; none exists. Pick well-known places
that fit their interests, give approximate coordinates, then call check_places_on_route to get the km marker and the
real distance from the route. Drop places that come back "far from route". List the rest in route order. For each:
name, near which town / km marker, extra detour, one line on why it is worth it, typical time needed, entry fee and
usual opening hours.
{fees}
Never invent a fee, an opening time or a toll. If you are not sure, write "not sure - check before you go".
Free places: say free. Label every fee "approx.".

STEP 4 - Practicalities. Call plan_stops (rest, fuel and, on long trips, overnight stops - name the town for each).
Call check_weather for the departure time and mention any warnings. If you know mileage and fuel price, call
estimate_trip_cost (ask for the fuel price once if missing; tolls only if the user gives them).
Vehicle notes: for a BIKE stress rain, wind, riding after dark, fatigue, helmet and riding gear, and remind the user
that the route may include roads closed to two-wheelers. For a CAR mention tolls and parking at popular places.

STEP 5 - Finish with a short day-by-day / stop-by-stop itinerary in route order, then offer export_trip
(Google Maps link + GPX file).

Style: plain text for a terminal - short lines, simple dashes for lists, no markdown tables. Distances in km. Times are
local. Be concise; do not repeat information the user already has."""


def ask_vehicle(input_fn=input, out=print) -> str:
    """Vehicle is asked first. The user just types car or bike (no numbered menu)."""
    while True:
        v = VEHICLE_ALIASES.get(input_fn("Assistant: Which vehicle are you travelling with - car or bike?\nYou: ").strip().lower())
        if v:
            return v
        out("Assistant: I plan trips for a car or a bike only. Please type car or bike.")


def _short(args: dict, n: int = 90) -> str:
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 3] + "..."


async def run_turn(llm, session, tools, messages, system, log=print) -> str:
    """One user message -> as many model/tool rounds as needed -> final text."""
    for _ in range(MAX_ROUNDS):
        resp = await llm.messages.create(model=MODEL, max_tokens=4096, system=system, tools=tools, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "pause_turn":       # a server-side tool (web search) is still working
            continue
        text = "".join(b.text for b in resp.content if b.type == "text")
        if resp.stop_reason != "tool_use":
            return text

        results = []
        for b in resp.content:
            if b.type != "tool_use":                # server_tool_use blocks are handled by the API itself
                continue
            log(f"  ... {b.name}({_short(b.input)})")
            try:
                out = await asyncio.wait_for(session.call_tool(b.name, b.input), TOOL_TIMEOUT)
                body = "\n".join(c.text for c in out.content if getattr(c, "type", "") == "text")
                is_error = bool(out.is_error)
            except Exception as e:                  # timeout / transport problem: tell the model, don't crash
                body, is_error = f"Tool call failed: {e!r}", True
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": body, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    return "I stopped after too many tool calls. Please narrow the request or ask again."


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")
    llm = AsyncAnthropic()
    env = {k: os.environ[k] for k in FORWARD_ENV if k in os.environ}
    params = StdioServerParameters(command=sys.executable, args=[str(HERE / "travel_server.py")], env=env)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                     for t in (await session.list_tools()).tools]
            if WEB_SEARCH:
                tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 6})
            system = build_system(WEB_SEARCH)
            messages: list = []

            async def turn(user_text: str) -> bool:
                mark = len(messages)
                messages.append({"role": "user", "content": user_text})
                try:
                    reply = await run_turn(llm, session, tools, messages, system)
                except Exception as e:              # e.g. API error: roll back so the history stays valid
                    del messages[mark:]
                    print(f"\n[error] {e}\nYour last message was not processed - please try again.\n")
                    return False
                print(f"\nAssistant: {reply}\n")
                return True

            vehicle = await asyncio.to_thread(ask_vehicle)
            if not await turn(f"I'm travelling by {vehicle}."):
                return
            while True:
                user = (await asyncio.to_thread(input, "You: ")).strip()
                if user.lower() in {"quit", "exit", "q"}:
                    break
                if user:
                    await turn(user)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
