# syntax=docker/dockerfile:1
FROM --platform=$BUILDPLATFORM golang:1.26-bookworm@sha256:dc40a13b61304647ed7e5d581d246b86413dfc29d63c6d018310feb93cd98fc0 AS build
ARG TARGETOS TARGETARCH
WORKDIR /src
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY *.go ./
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH go build -trimpath -ldflags="-s -w" -o /modstore .

FROM scratch
COPY --from=build /modstore /modstore
USER 65532:65532
ENV MODSTORE_DATA_DIR=/data MODSTORE_PUBLIC_ADDR=0.0.0.0:8767 GOMEMLIMIT=120MiB
EXPOSE 8767
ENTRYPOINT ["/modstore"]
CMD ["serve"]
