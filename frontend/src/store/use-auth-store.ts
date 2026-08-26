// src/store/use-auth-store.ts
import {
  create,
} from "zustand";

import {
  ApiError,
  fetchCurrentUser,
  getAuthToken,
  loginAccount,
  logoutAccount,
  type AuthUser,
} from "@/lib/api";

const AUTH_TOKEN_KEY =
  "dga-auth-token";

interface AuthState {
  user: AuthUser | null;
  status:
    | "loading"
    | "authenticated"
    | "unauthenticated";
  error: string | null;

  init: () => Promise<void>;
  login: (
    email: string,
    password: string
  ) => Promise<void>;
  logout: () => Promise<void>;
  clearError: () => void;
}

function storeToken(
  token: string
) {
  window.localStorage.setItem(
    AUTH_TOKEN_KEY,
    token
  );
}

function clearToken() {
  window.localStorage.removeItem(
    AUTH_TOKEN_KEY
  );
}

export const useAuthStore =
  create<AuthState>(
    (set, get) => ({
      user: null,
      status: "loading",
      error: null,

      init: async () => {
        if (
          get().status ===
          "authenticated"
        ) {
          return;
        }

        if (!getAuthToken()) {
          set({
            status:
              "unauthenticated",
            user: null,
          });

          return;
        }

        const user =
          await fetchCurrentUser();

        if (user) {
          set({
            status:
              "authenticated",
            user,
            error: null,
          });
        } else {
          clearToken();

          set({
            status:
              "unauthenticated",
            user: null,
          });
        }
      },

      login: async (
        email,
        password
      ) => {
        set({
          error: null,
        });

        try {
          const {
            user,
            token,
          } =
            await loginAccount(
              email,
              password
            );

          storeToken(token);

          set({
            user,
            status:
              "authenticated",
            error: null,
          });
        } catch (error) {
          const message =
            error instanceof ApiError
              ? error.message
              : "Login failed.";

          set({
            error: message,
          });

          throw error;
        }
      },

      logout: async () => {
        await logoutAccount();

        clearToken();

        set({
          user: null,
          status:
            "unauthenticated",
          error: null,
        });
      },

      clearError: () =>
        set({
          error: null,
        }),
    })
  );