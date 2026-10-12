"use client";

import { FormEvent, useEffect, useState } from "react";

type SourceItem = {
  source: string;
  page: number;
  text: string;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [documents, setDocuments] = useState<string[]>([]);
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadDocuments = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/documents`);
        if (!response.ok) {
          throw new Error("Unable to fetch document list.");
        }
        const data = await response.json();
        setDocuments(data.documents ?? []);
      } catch {
        setDocuments([]);
      }
    };

    loadDocuments();
  }, []);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();

    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/api/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Unable to retrieve an answer.");
      }

      setAnswer(data.answer ?? "No answer found.");
      setSources(data.sources ?? []);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-10 text-slate-100">
      <div className="mx-auto max-w-6xl">
        <header className="mb-8 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.25em] text-cyan-300">RAG Assistant</p>
            <h1 className="mt-2 text-3xl font-bold md:text-5xl">Ask questions from your PDFs</h1>
          </div>
          <div className="rounded-full border border-cyan-500/40 bg-cyan-500/10 px-3 py-1 text-sm text-cyan-200">
            {documents.length} document(s) indexed
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <section className="rounded-3xl border border-slate-800 bg-slate-900/80 p-5 shadow-2xl shadow-slate-950/40">
            <form onSubmit={handleSubmit} className="space-y-4">
              <label htmlFor="question" className="block text-sm font-medium text-slate-300">
                Your question
              </label>
              <textarea
                id="question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                rows={6}
                placeholder="Ask about the contents of your PDF collection..."
                className="w-full rounded-2xl border border-slate-700 bg-slate-950 p-4 text-base text-slate-100 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none"
              />
              <div className="flex items-center justify-between gap-3">
                <button
                  type="submit"
                  disabled={loading}
                  className="rounded-full bg-cyan-500 px-5 py-2.5 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300"
                >
                  {loading ? "Thinking..." : "Ask the agent"}
                </button>
                {error ? <span className="text-sm text-rose-400">{error}</span> : null}
              </div>
            </form>

            <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-950/80 p-4">
              <h2 className="mb-3 text-lg font-semibold text-slate-200">Answer</h2>
              <p className="whitespace-pre-wrap leading-7 text-slate-200">
                {answer || "Ask a question to retrieve the answer from the indexed documents."}
              </p>
            </div>
          </section>

          <aside className="space-y-6">
            <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-5">
              <h2 className="mb-4 text-lg font-semibold text-slate-200">Available documents</h2>
              <ul className="space-y-2 text-sm text-slate-300">
                {documents.length > 0 ? (
                  documents.map((document) => (
                    <li key={document} className="rounded-xl border border-slate-800 bg-slate-950/80 px-3 py-2">
                      {document}
                    </li>
                  ))
                ) : (
                  <li className="text-slate-500">No PDFs found in the knowledge base yet.</li>
                )}
              </ul>
            </div>

            <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-5">
              <h2 className="mb-4 text-lg font-semibold text-slate-200">Source references</h2>
              {sources.length > 0 ? (
                <div className="space-y-3">
                  {sources.map((source, index) => (
                    <div key={`${source.source}-${source.page}-${index}`} className="rounded-xl border border-slate-800 bg-slate-950/80 p-3">
                      <p className="text-sm font-medium text-cyan-300">
                        {source.source} · Page {source.page}
                      </p>
                      <p className="mt-2 line-clamp-5 text-sm text-slate-300">{source.text}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-500">Relevant citations will appear here after a question is asked.</p>
              )}
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}
