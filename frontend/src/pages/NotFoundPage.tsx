import { Link } from 'react-router-dom'

export default function NotFoundPage() {
  return (
    <div className="mx-auto max-w-md py-20 text-center">
      <p className="text-5xl font-semibold text-brand-600">404</p>
      <h1 className="mt-3 text-xl font-semibold">Page not found</h1>
      <p className="mt-2 text-sm text-slate-500">
        The page you were looking for does not exist or has moved.
      </p>
      <Link to="/" className="btn-primary mt-6">Back to home</Link>
    </div>
  )
}
