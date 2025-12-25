import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";

export default function Home() {
  return (
    <Layout
      title="Physical AI & Humanoid Robotics"
      description="An AI-native textbook for humanoid robotics, ROS 2, simulation, and embodied intelligence."
    >
      {/* TITLE PAGE */}
      <header style={styles.header}>
        <h1 style={styles.title}>
          Physical AI & Humanoid Robotics
        </h1>

        <p style={styles.subtitle}>
          An AI-native textbook for humanoid robotics, ROS 2, simulation,
          Vision-Language-Action systems, and embodied intelligence.
        </p>

        <Link to="/ docs/intro" style={styles.primaryLink}>
          Read the Introduction →
        </Link>
      </header>

      {/* MAIN CONTENT */}
      <main style={styles.main}>
        {/* PREFACE */}
        <section style={styles.section}>
          <h2 style={styles.heading}>Preface</h2>
          <p style={styles.text}>
            This textbook is designed for engineers, researchers, and advanced
            learners who want to build intelligent machines operating in the
            physical world. It focuses on modern robotics pipelines combining
            simulation, hardware, and AI-native systems.
          </p>
        </section>

        {/* CONTENTS */}
        

        {/* AUDIENCE */}
        <section style={styles.section}>
          <h2 style={styles.heading}>Who This Book Is For</h2>
          <p style={styles.text}>
            This book is intended for readers with foundational knowledge of
            programming and linear algebra. No prior robotics experience is
            required, but curiosity and discipline are essential.
          </p>
        </section>
      </main>

      {/* FOOTER CTA */}
      <footer style={styles.footer}>
        <Link to="/docs/intro">
          Start Reading →
        </Link>
      </footer>
    </Layout>
  );
}

/* ===================== */
/* DARK TEXTBOOK STYLES */
/* ===================== */

const styles = {
  header: {
    padding: "90px 20px 60px",
    textAlign: "center",
    backgroundColor: "#0f1115",
    borderBottom: "1px solid #1f2937",
  },

  title: {
    fontSize: "44px",
    fontWeight: "600",
    marginBottom: "18px",
    color: "#e5e7eb",
  },

  subtitle: {
    fontSize: "18px",
    maxWidth: "720px",
    margin: "0 auto 30px",
    lineHeight: "1.7",
    color: "#9ca3af",
  },

  primaryLink: {
    fontSize: "18px",
    color: "#93c5fd",
    textDecoration: "none",
  },

  main: {
    maxWidth: "760px",
    margin: "0 auto",
    padding: "60px 20px",
  },

  section: {
    marginBottom: "60px",
  },

  heading: {
    fontSize: "26px",
    fontWeight: "500",
    marginBottom: "18px",
    color: "#f3f4f6",
  },

  text: {
    fontSize: "17px",
    lineHeight: "1.8",
    color: "#d1d5db",
  },

  list: {
    paddingLeft: "22px",
    fontSize: "17px",
    lineHeight: "1.9",
  },

  footer: {
    padding: "50px 20px",
    textAlign: "center",
    borderTop: "1px solid #1f2937",
    color: "#9ca3af",
  },
};
