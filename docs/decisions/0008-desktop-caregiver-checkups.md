# 0008 — Desktop "house", caregiver, and checkups

- **Date:** 2026-10-19
- **Status:** accepted (user: "I love the caregiver"; desktop and checkups raised no objection)
- **Decided by:** user, on agent proposal
- **Resolves:** Q2 (partially), Q5 (partially)

## Decision
- **World structure:** an OS-like desktop shell with a persistent taskbar/dock. "Rooms" are apps and windows within it: crib, toys, hide-and-seek, places, picture books, tools, small games, mini-web. The physics (cursor, click, scroll, keys) is identical everywhere. Rooms are procedurally generated so every milestone has fresh variants. Content is public domain or openly licensed.
- **Caregiver:** a harness component that moves the learner between rooms early on, based on learning progress (move on when learning stalls or the room is mastered; go where learning is likely fastest). Its rule is a shared, improvable setting: a meta-learning landscape in its own right. Every caregiver move is logged as a harness action. Once the learner reliably uses the taskbar itself (a milestone), the caregiver steps back to a safety net.
- **Measurement:** milestones observed during everyday life are logged. **Checkups** (fresh, never-seen variants inserted briefly by the harness) are the score that counts.
- **Rendering** is an implementation detail, invisible to the learner. Use whatever is fastest to get working.
