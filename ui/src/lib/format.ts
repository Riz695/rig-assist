// Numbers as the UI shows them.

export function formatSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  const whole = Math.round(seconds)
  return `${Math.floor(whole / 60)} min ${whole % 60} s`
}

export function formatTokens(tokens: number): string {
  return `${tokens.toLocaleString('en-US')} ${tokens === 1 ? 'token' : 'tokens'}`
}

/** null means LLM prices aren't configured: say so instead of showing $0. */
export function formatCost(usd: number | null | undefined): string {
  if (usd === null || usd === undefined) return 'cost unknown'
  return `$${usd.toFixed(usd < 1 ? 4 : 2)}` // a job costs fractions of a cent to cents
}
