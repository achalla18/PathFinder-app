import { useState } from "react";
import { type FeedbackAnswers, sendFeedback } from "./api";

type Answer = "yes" | "partly" | "no";
const QUESTIONS: [keyof Omit<FeedbackAnswers, "walked" | "comment">, string][] = [
  ["felt_minutes_clear", "Did “feels easier / harder” in minutes make sense?"],
  ["places_clear", "Were the numbered decision points clear?"],
  ["better_than_route_list", "Is this better than a plain list of routes?"],
];
const ANSWERS: [Answer, string][] = [["yes", "Yes"], ["partly", "Partly"], ["no", "No"]];

/** A tester's verdict, in three taps and a sentence. Nothing is sent until they press Send. */
export function Feedback({ walked, onClose }: { walked: boolean; onClose: () => void }) {
  const [answers, setAnswers] = useState<Partial<Record<string, Answer>>>({});
  const [comment, setComment] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | "failed">("idle");
  const empty = Object.keys(answers).length === 0 && !comment.trim();

  async function send() {
    setState("sending");
    try {
      await sendFeedback({ ...answers, walked, comment: comment.trim() });
      setState("sent");
    } catch {
      setState("failed");
    }
  }

  if (state === "sent") {
    return <div className="place feedback"><h4>Thank you<button className="link" onClick={onClose}>close</button></h4><p className="hint">Your answers were saved.</p></div>;
  }
  return (
    <div className="place feedback">
      <h4>{walked ? "How was that walk?" : "How is this working for you?"}<button className="link" onClick={onClose}>close</button></h4>
      {QUESTIONS.map(([key, label]) => (
        <div key={key} className="question">
          <p>{label}</p>
          <div className="chips segmented">
            {ANSWERS.map(([v, text]) => (
              <button key={v} className={answers[key] === v ? "chip on" : "chip"} aria-pressed={answers[key] === v}
                      onClick={() => setAnswers((a) => ({ ...a, [key]: v }))}>{text}</button>
            ))}
          </div>
        </div>
      ))}
      <textarea aria-label="Anything confusing, wrong or missing?" placeholder="Anything confusing, wrong or missing?" maxLength={2000} rows={3}
                value={comment} onChange={(e) => setComment(e.target.value)} />
      <button className="btn primary" disabled={empty || state === "sending"} onClick={send}>{state === "sending" ? "Sending…" : "Send"}</button>
      {state === "failed" && <p className="error">That did not go through. Try again in a moment.</p>}
      <p className="fine">Only these answers are saved, with today’s date: no location, no trip, nothing about your phone.</p>
    </div>
  );
}
