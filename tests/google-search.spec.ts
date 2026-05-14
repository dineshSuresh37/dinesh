import { test, expect } from '@playwright/test';

test.describe('Google Search', () => {
  test('search for TN election 2026', async ({ page }) => {
    await page.goto('https://www.google.com');

    await page.getByRole('combobox', { name: 'Search' }).fill('TN election 2026');
    await page.getByRole('combobox', { name: 'Search' }).press('Enter');

    await expect(page).toHaveURL(/search\?q=TN\+election\+2026/);
    await expect(page.getByRole('heading', { name: /TN election 2026/i }).or(
      page.locator('#search')
    )).toBeVisible();
  });

  test('google logo is visible on homepage', async ({ page }) => {
    await page.goto('https://www.google.com');

    const logo = page.locator('img[alt="Google"]');
    await expect(logo).toBeVisible();
  });
});
