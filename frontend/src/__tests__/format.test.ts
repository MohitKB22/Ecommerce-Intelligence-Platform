import { describe, expect, it } from 'vitest'
import { compactNumber, currency, percent, relativeTime, shortDate, titleCase } from '@/utils/format'

describe('formatting helpers', () => {
  it('formats currency', () => {
    expect(currency(1234.5)).toBe('$1,234.50')
    expect(currency(0)).toBe('$0.00')
  })

  it('survives non-finite input rather than rendering NaN', () => {
    expect(currency(Number.NaN)).toBe('$0.00')
    expect(percent(Number.POSITIVE_INFINITY)).toBe('0.0%')
    expect(compactNumber(Number.NaN)).toBe('0')
  })

  it('formats percentages with configurable precision', () => {
    expect(percent(0.1234)).toBe('12.3%')
    expect(percent(0.1234, 2)).toBe('12.34%')
  })

  it('compacts large numbers', () => {
    expect(compactNumber(1_500_000)).toBe('1.5M')
    expect(compactNumber(2400)).toBe('2.4K')
  })

  it('title-cases snake and kebab case', () => {
    expect(titleCase('battery_life')).toBe('Battery Life')
    expect(titleCase('high-value')).toBe('High Value')
  })

  it('handles missing or malformed dates', () => {
    expect(shortDate(null)).toBe('-')
    expect(shortDate('not-a-date')).toBe('-')
    expect(relativeTime(undefined)).toBe('never')
  })
})
