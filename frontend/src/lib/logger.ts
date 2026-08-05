export function ts() {
  return new Date().toISOString()
}

export function info(...args: unknown[]) {
  console.info(ts(), '[INFO]', ...args)
}
export function debug(...args: unknown[]) {
  console.debug(ts(), '[DEBUG]', ...args)
}
export function warn(...args: unknown[]) {
  console.warn(ts(), '[WARN]', ...args)
}
export function error(...args: unknown[]) {
  console.error(ts(), '[ERROR]', ...args)
}

// Optional: later extend to POST logs to backend for collection
const logger = { info, debug, warn, error }
export default logger
