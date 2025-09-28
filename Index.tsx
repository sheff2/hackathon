import { useState } from 'react';
import RouteSearchInterface from '@/components/RouteSearchInterface';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Shield, Navigation, Brain, Users, Info } from 'lucide-react';
import { Link } from 'react-router-dom';
import { ask, startSession } from '../../adkclient';  

interface ParsedRouteDetails {
  origin?: string;
  destination?: string;
  waypoints?: string[];
}


type RouteItem = {
  id: number;
  duration: string;
  distance: string;
  risk: string;
  link: string;
};

function parseGoogleMapsLink(link: string): ParsedRouteDetails {
  const normalise = (value?: string | null) =>
    value ? value.replace(/\+/g, ' ').trim() : undefined;
  try {
    const url = new URL(link);
    const params = new URLSearchParams(url.search);

    let destination = normalise(params.get('destination'));
    let origin = normalise(params.get('origin'));
    const waypointParam = params.get('waypoints');
    const waypoints = waypointParam
      ? waypointParam
          .split('|')
          .map((item) => normalise(item))
          .filter((item): item is string => Boolean(item))
      : undefined;

    if (!destination) {
      const segments = url.pathname
        .split('/')
        .map((segment) => normalise(decodeURIComponent(segment)))
        .filter((segment): segment is string => Boolean(segment));
      const dirIndex = segments.findIndex(
        (segment) => segment.toLowerCase() === 'dir'
      );
      if (dirIndex >= 0) {
        const possible = segments.slice(dirIndex + 1);
        if (!origin && possible[0]) origin = possible[0];
        if (!destination && possible[1]) destination = possible[1];
      }
    }

    return { origin, destination, waypoints };
  } catch {
    return {};
  }
}

function buildEmbedDirectionsUrl(details: ParsedRouteDetails, apiKey: string) {
  if (!details.destination) return null;

  const params = new URLSearchParams({
    key: apiKey,
    origin: details.origin ?? 'Current Location',
    destination: details.destination,
    mode: 'walking',
  });

  if (details.waypoints?.length) {
    params.set('waypoints', details.waypoints.join('|'));
  }

  return `https://www.google.com/maps/embed/v1/directions?${params.toString()}`;
}

// function parseRoutesResponse(text: string) {
//   // Match each numbered route with Duration, Distance, Risk Summary, Maps Link
//   const routeRegex =
//     /(\d+)\.\s\*\*Duration:\*\*\s([^,]+),\s\*\*Distance:\*\*\s([^,]+),\s\*\*Risk Summary:\*\*\s(.+?),\s\*\*Maps Link:\*\*\s(\S+)/g;

//   const routes: {
//     id: number;
//     duration: string;
//     distance: string;
//     risk: string;
//     link: string;
//   }[] = [];

//   let match;
//   while ((match = routeRegex.exec(text)) !== null) {
//     routes.push({
//       id: Number(match[1]),
//       duration: match[2].trim(),
//       distance: match[3].trim(),
//       risk: match[4].trim(),
//       link: match[5].trim(),
//     });
//   }

//   return routes;
// }

function parseRoutesResponse(text: string) {
  // s = dotAll, so . matches across newlines
  // g = global, so it finds all routes
  const routeRegex =
    /(\d+)\.\s*\*\*Duration:\*\*\s*([^,]+),\s*\*\*Distance:\*\*\s*([^,]+),\s*\*\*Risk Summary:\*\*\s*([\s\S]*?)\s*\*\*Maps Link:\*\*\s*(\S+?)(?=["\s]|$)/gs;

  type RouteItem = {
    id: number;
    duration: string;
    distance: string;
    risk: string;
    link: string;
  };

  const routes: RouteItem[] = [];
  let match: RegExpExecArray | null;

  while ((match = routeRegex.exec(text)) !== null) {
    const rawLink = match[5].trim();
    const cleanLink = rawLink.replace(/[",]+$/g, ""); // strip quotes or commas

    routes.push({
      id: Number(match[1]),
      duration: match[2].trim(),
      distance: match[3].trim(),
      risk: match[4].trim().replace(/\s+/g, " "), // normalize newlines
      link: cleanLink,
    });
  }

  return routes;
}



export default function Index() {
  const [isSearching, setIsSearching] = useState(false);
  const [currentRouteLink, setCurrentRouteLink] = useState<string | null>(null);
  const [currentRouteDetails, setCurrentRouteDetails] =
    useState<ParsedRouteDetails | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [routes, setRoutes] = useState<RouteItem[] | null>(null);


  const embedApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string;
  const embedSrc =
    currentRouteDetails && embedApiKey
      ? buildEmbedDirectionsUrl(currentRouteDetails, embedApiKey)
      : null;

  const handleRouteSearch = async (userText: string) => {
    setIsSearching(true);
    setError(null);
    try {
      // Replace with your agent logic; demo hard-coded link:
      // Call your local agent
      // await startSession();
      // const resp = await ask(userText);

      //harcoded response for demo purposes
      const resp = "Here you go — I generated 5 routes and ranked their safety. 1. **Duration:** 297 min, **Distance:** 21.34 km, **Risk Summary:** Domestic Violence on Mar 24, 2024 (about 127 m away); Aggravated Assault on Mar 17, 2024 (about 6 m away); Aggravated Assault on Mar 31, 2024 (about 66 m away); plus 20 other incidents within 500 m, **Maps Link:** https://www.google.com/maps/dir/?api=1&origin=25.713590000000003,-80.27922000000001&destination=25.79071,-80.13006&travelmode=walking&waypoints=25.713580000000004,-80.2792|25.724030000000003,-80.2663|25.73195,-80.25606|25.737140000000004,-80.24602|25.743250000000003,-80.23815|25.7494,-80.23838|25.75411,-80.23854|25.76151,-80.23884000000001|25.76846,-80.23521000000001|25.772610000000004,-80.2262|25.77342,-80.21456|25.773950000000003,-80.20437000000001|25.775050000000004,-80.1937|25.782890000000002,-80.19383|25.786880000000004,-80.19059|25.790090000000003,-80.17789|25.790570000000002,-80.17008000000001|25.79106,-80.15861000000001|25.79156,-80.14647000000001|25.792080000000002,-80.13649000000001 2. **Duration:** 417 min, **Distance:** 30.06 km, **Risk Summary:** Domestic Violence on Mar 24, 2024 (about 131 m away); Simple Assault on Apr 02, 2024 (about 67 m away); Motor Vehicle Theft on Mar 15, 2024 (about 94 m away); plus 28 other incidents within 500 m, **Maps Link:** https://www.google.com/maps/dir/?api=1&origin=25.713590000000003,-80.27922000000001&destination=25.790730000000003,-80.13006&travelmode=walking&waypoints=25.713580000000004,-80.2792|25.72538,-80.26464|25.732380000000003,-80.25577000000001|25.737940000000002,-80.24287000000001|25.741930000000004,-80.23194000000001|25.74436,-80.22103000000001|25.749450000000003,-80.20351000000001|25.746190000000002,-80.19121000000001|25.7447,-80.17526000000001|25.745870000000004,-80.19816|25.748610000000003,-80.21047|25.75354,-80.20534|25.764190000000003,-80.19135|25.772420000000004,-80.19012000000001|25.78188,-80.19063000000001|25.78901,-80.18744000000001|25.790280000000003,-80.17434|25.790840000000003,-80.16249|25.79146,-80.14915|25.792070000000002,-80.13671000000001 3. **Duration:** 293 min, **Distance:** 20.89 km, **Risk Summary:** Domestic Violence on Mar 24, 2024 (about 129 m away); Simple Assault on Apr 02, 2024 (about 67 m away); Motor Vehicle Theft on Mar 15, 2024 (about 88 m away); plus 28 other incidents within 500 m, **Maps Link:** https://www.google.com/maps/dir/?api=1&origin=25.713590000000003,-80.27922000000001&destination=25.790730000000003,-80.13006&travelmode=walking&waypoints=25.713580000000004,-80.2792|25.72323,-80.26728|25.73017,-80.25850000000001|25.73522,-80.25219000000001|25.73833,-80.24183000000001|25.741200000000003,-80.23393|25.74592,-80.22185|25.75027,-80.21618000000001|25.7529,-80.20877|25.755440000000004,-80.19911|25.76151,-80.19338|25.768400000000003,-80.19031000000001|25.77417,-80.19032|25.78215,-80.1906|25.788130000000002,-80.18789000000001|25.79024,-80.17611000000001|25.790770000000002,-80.1697|25.791,-80.15944|25.79147,-80.14916000000001|25.792,-80.13996"
      const match = resp.match(/https:\/\/www\.google\.com\/maps\/dir\/\?api=1[^\s]*/);
      const link = match ? match[0] : null;

      if (!link) throw new Error('Agent response did not include a maps link.');

      const parsed = parseRoutesResponse(resp);
      setRoutes(parsed);

      setCurrentRouteLink(link);
      setCurrentRouteDetails(parseGoogleMapsLink(link));

      // If you want to draw the route on your map component, pass only destination
      // (no current location). Remove any code that tried to read navigator.geolocation.
      const calc = (window as any).calculateSafeRoute as
        | ((dest: string, origin?: string, opts?: { waypoints?: { location: string; stopover?: boolean }[] }) => void)
        | undefined;

      const details = parseGoogleMapsLink(link);
      if (calc && details.destination) {
        const mappedWaypoints = details.waypoints?.map((w) => ({ location: w }));
        // origin intentionally omitted to avoid using current location
        calc(details.destination, undefined, {
          waypoints: mappedWaypoints && mappedWaypoints.length ? mappedWaypoints : undefined,
        });
      }
    } catch (e) {
      console.error(e);
      setError(e instanceof Error ? e.message : 'Failed to fetch route.');
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <div className="container mx-auto px-4 py-6">
        <div className="text-center mb-8">
          <div className="flex justify-between items-start mb-4">
            <div className="flex-1" />
            <div className="inline-flex items-center gap-3">
              <div className="p-2 bg-gradient-primary rounded-lg shadow-glow">
                <Shield className="w-6 h-6 text-primary-foreground" />
              </div>
              <h1 className="text-3xl md:text-5xl font-bold bg-gradient-to-r from-primary via-primary-glow to-accent bg-clip-text text-transparent">
                Home Safe
              </h1>
            </div>
            <div className="flex-1 flex justify-end">
              <Link to="/about">
                <Button variant="outline" size="sm" className="gap-2">
                  <Info className="w-4 h-4" />
                  About
                </Button>
              </Link>
            </div>
          </div>
          <p className="text-lg text-muted-foreground max-w-xl mx-auto">
            AI-powered route optimization for your safest journey home
          </p>
        </div>

        {/* Search Interface */}
        <div className="max-w-2xl mx-auto mb-8">
          <RouteSearchInterface
            onSearchRoute={handleRouteSearch}
            isLoading={isSearching}
            currentRoute={
              currentRouteDetails
                ? {
                    origin: currentRouteDetails.origin ?? 'Current Location',
                    destination:
                      currentRouteDetails.destination ?? 'Selected Destination',
                    link: currentRouteLink ?? undefined,
                  }
                : null
            }
            error={error}
          />
        </div>

      {/* Routes Row (dynamic) */}
      <div className="max-w-4xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {routes?.map((route, i) => (
            <Card key={route.id} className="p-4 bg-card border shadow-glass">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-primary/10 rounded-lg shrink-0">
                  {i === 0 && <Brain className="w-5 h-5 text-primary" />}
                  {i === 1 && <Navigation className="w-5 h-5 text-accent" />}
                  {i >= 2 && <Users className="w-5 h-5 text-accent" />}
                </div>

                <div className="space-y-2">
                  <h3 className="font-semibold text-foreground">
                    Route {route.id}
                  </h3>

                  {/* duration + distance as your subtitle line */}
                  <p className="text-sm text-muted-foreground">
                    {route.duration} • {route.distance}
                  </p>

                  {/* full safety/risk context from the response */}
                  <div className="text-sm leading-relaxed">
                    <span className="font-medium">Risk summary:</span>{" "}
                    {route.risk}
                  </div>

                  {/* open the exact maps link from the response */}
                  <div className="pt-2">
                    <Button asChild size="sm">
                      <a href={route.link} target="_blank" rel="noopener noreferrer">
                        Open in Google Maps
                      </a>
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
        

        {/* Embedded Google Maps Directions (only map you see now) */}
        {embedSrc && (
          <div className="max-w-6xl mx-auto mb-8">
            <Card className="p-2 bg-card border shadow-glass">
              <div className="rounded-lg overflow-hidden bg-white">
                <iframe
                  title="Embedded Directions"
                  src={embedSrc}
                  width="100%"
                  height="600"
                  style={{ border: 0, backgroundColor: '#fff' }}
                  loading="lazy"
                  referrerPolicy="no-referrer-when-downgrade"
                  allowFullScreen
                />
              </div>
              <div className="p-3 flex justify-end">
                {currentRouteLink && (
                  <a
                    href={currentRouteLink}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Button variant="default">Open in Google Maps</Button>
                  </a>
                )}
              </div>
            </Card>
          </div>
        )}

        {/* Feature cards */}
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="p-4 bg-card border shadow-glass">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <Brain className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">AI Analysis</h3>
                  <p className="text-sm text-muted-foreground">
                    Real-time safety assessment
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-4 bg-card border shadow-glass">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-accent/10 rounded-lg">
                  <Navigation className="w-5 h-5 text-accent" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">Smart Routes</h3>
                  <p className="text-sm text-muted-foreground">
                    Optimized for safety
                  </p>
                </div>
              </div>
            </Card>
            <Card className="p-4 bg-card border shadow-glass">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-accent/10 rounded-lg">
                  <Users className="w-5 h-5 text-accent" />
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">24/7 Monitoring</h3>
                  <p className="text-sm text-muted-foreground">Always up to date</p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}