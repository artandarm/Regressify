import type { NextRequest } from "next/server";

const BACKEND = "http://localhost:8000";

export const dynamic = "force-dynamic";

const HOP_BY_HOP = new Set(["host", "connection", "transfer-encoding", "keep-alive", "te", "trailer", "upgrade"]);

async function proxy(req: NextRequest, params: { path: string[] }): Promise<Response> {
  const path = params.path.join("/");
  const target = `${BACKEND}/${path}${req.nextUrl.search}`;

  const fwdHeaders = new Headers();
  req.headers.forEach((val, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) fwdHeaders.set(key, val);
  });

  const hasBody = req.method !== "GET" && req.method !== "HEAD";
  const body = hasBody ? await req.arrayBuffer() : undefined;

  const upstream = await fetch(target, {
    method: req.method,
    headers: fwdHeaders,
    body: body && body.byteLength > 0 ? body : undefined,
  });

  const resHeaders = new Headers();
  upstream.headers.forEach((val, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) resHeaders.set(key, val);
  });

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: resHeaders,
  });
}

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(req: NextRequest, { params }: Ctx) {
  return proxy(req, await params);
}

export async function POST(req: NextRequest, { params }: Ctx) {
  return proxy(req, await params);
}

export async function PUT(req: NextRequest, { params }: Ctx) {
  return proxy(req, await params);
}

export async function DELETE(req: NextRequest, { params }: Ctx) {
  return proxy(req, await params);
}
