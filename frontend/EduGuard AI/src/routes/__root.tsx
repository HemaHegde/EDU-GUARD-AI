import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import {
  Outlet,
  Link,
  createRootRouteWithContext,
  useRouter,
  HeadContent,
  Scripts,
  useNavigate,
} from "@tanstack/react-router";

import {
  useEffect,
  useState,
  type ReactNode,
} from "react";

import appCss from "../styles.css?url";

import { reportLovableError } from "../lib/lovable-error-reporting";

import { Sidebar } from "@/components/eg/Sidebar";

import { AnimatedBackground } from "@/components/eg/AnimatedBackground";

import { supabase } from "@/lib/supabase";

// =========================
// NOT FOUND
// =========================

function NotFoundComponent() {

  return (

    <div className="flex min-h-screen items-center justify-center bg-background px-4">

      <div className="max-w-md text-center">

        <h1 className="text-7xl font-bold text-foreground">
          404
        </h1>

        <h2 className="mt-4 text-xl font-semibold text-foreground">
          Page not found
        </h2>

        <p className="mt-2 text-sm text-muted-foreground">

          The page you're looking for doesn't exist or has been moved.

        </p>

        <div className="mt-6">

          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >

            Go home

          </Link>

        </div>

      </div>

    </div>

  );

}

// =========================
// ERROR COMPONENT
// =========================

function ErrorComponent({

  error,
  reset,

}: {

  error: Error;
  reset: () => void;

}) {

  console.error(error);

  const router = useRouter();

  useEffect(() => {

    reportLovableError(

      error,

      {
        boundary:
          "tanstack_root_error_component",
      }

    );

  }, [error]);

  return (

    <div className="flex min-h-screen items-center justify-center bg-background px-4">

      <div className="max-w-md text-center">

        <h1 className="text-xl font-semibold tracking-tight text-foreground">

          This page didn't load

        </h1>

        <p className="mt-2 text-sm text-muted-foreground">

          Something went wrong on our end.
          You can try refreshing or head back home.

        </p>

        <div className="mt-6 flex flex-wrap justify-center gap-2">

          <button

            onClick={() => {

              router.invalidate();

              reset();

            }}

            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"

          >

            Try again

          </button>

          <a

            href="/"

            className="inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"

          >

            Go home

          </a>

        </div>

      </div>

    </div>

  );

}

// =========================
// ROOT ROUTE
// =========================

export const Route =
  createRootRouteWithContext<{

    queryClient: QueryClient;

  }>()({

    head: () => ({

      meta: [

        {
          charSet: "utf-8",
        },

        {
          name: "viewport",
          content:
            "width=device-width, initial-scale=1",
        },

        {
          title: "EduGuard-AI",
        },

        {
          name: "description",

          content:
            "AI-powered emotional and educational support for every learner.",
        },

      ],

      links: [

        {
          rel: "stylesheet",
          href: appCss,
        },

      ],

    }),

    shellComponent: RootShell,

    component: RootComponent,

    notFoundComponent:
      NotFoundComponent,

    errorComponent:
      ErrorComponent,

  });

// =========================
// ROOT SHELL
// =========================

function RootShell({

  children,

}: {

  children: ReactNode;

}) {

  return (

    <html lang="en">

      <head>

        <HeadContent />

      </head>

      <body>

        {children}

        <Scripts />

      </body>

    </html>

  );

}

// =========================
// ROOT COMPONENT
// =========================

function RootComponent() {

  const { queryClient } =
    Route.useRouteContext();

  const navigate =
    useNavigate();

  const [loading, setLoading] =
    useState(true);

  const [authenticated,
    setAuthenticated] =
    useState(false);

  // =========================
  // CHECK AUTH
  // =========================

  useEffect(() => {

    checkSession();

  }, []);

  async function checkSession() {

    const {

      data: { session },

    } = await supabase.auth.getSession();

    // -------------------------
    // NOT LOGGED IN
    // -------------------------

    if (!session) {

      setAuthenticated(false);

      const path =
        window.location.pathname;

      // ALLOW LOGIN + SIGNUP

      if (

        path !== "/login" &&
        path !== "/signup"

      ) {

        navigate({
          to: "/login",
        });

      }

    }

    else {

      setAuthenticated(true);

    }

    setLoading(false);

  }

  // =========================
  // LOADING SCREEN
  // =========================

  if (loading) {

    return (

      <div className="flex min-h-screen items-center justify-center bg-background">

        <div className="text-lg font-semibold">

          Loading EduGuard-AI...

        </div>

      </div>

    );

  }

  // =========================
  // AUTH PAGES
  // =========================

  const path =
    window.location.pathname;

  const isAuthPage =

    path === "/login" ||
    path === "/signup";

  // =========================
  // LOGIN / SIGNUP
  // =========================

  if (isAuthPage) {

    return (

      <QueryClientProvider
        client={queryClient}
      >

        <Outlet />

      </QueryClientProvider>

    );

  }

  // =========================
  // MAIN APP
  // =========================

  return (

    <QueryClientProvider
      client={queryClient}
    >

      <div className="relative flex min-h-screen w-full">

        <AnimatedBackground />

        <Sidebar />

        <main className="min-w-0 flex-1">

          <Outlet />

        </main>

      </div>

    </QueryClientProvider>

  );

}