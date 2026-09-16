import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EmptyState, ErrorState, Rating, Stat, Tabs } from '@/components/ui'

describe('UI primitives', () => {
  it('renders an error state and calls the retry handler', async () => {
    const onRetry = vi.fn()
    const user = userEvent.setup()
    render(<ErrorState title="Search failed" message="Backend unreachable" onRetry={onRetry} />)
    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByText('Backend unreachable')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /try again/i }))
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('renders an empty state with an action', () => {
    render(<EmptyState title="No products" message="Try another filter" action={<button>Reset</button>} />)
    expect(screen.getByText('No products')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reset' })).toBeInTheDocument()
  })

  it('exposes an accessible rating label', () => {
    render(<Rating value={4.25} count={99} />)
    expect(screen.getByLabelText('Rated 4.3 out of 5')).toBeInTheDocument()
    expect(screen.getByText('4.3 (99)')).toBeInTheDocument()
  })

  it('shows positive and negative change on a stat', () => {
    const { rerender } = render(<Stat label="Revenue" value="$1,000" change={12.5} />)
    expect(screen.getByText('+12.5%')).toBeInTheDocument()
    rerender(<Stat label="Revenue" value="$1,000" change={-4.25} />)
    expect(screen.getByText('-4.3%')).toBeInTheDocument()
  })

  it('switches tabs', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(
      <Tabs tabs={[{ id: 'a', label: 'Overview' }, { id: 'b', label: 'Reviews' }]} active="a" onChange={onChange} />,
    )
    expect(screen.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true')
    await user.click(screen.getByRole('tab', { name: 'Reviews' }))
    expect(onChange).toHaveBeenCalledWith('b')
  })
})
