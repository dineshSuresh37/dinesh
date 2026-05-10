import { test, expect } from '@playwright/test';

test.describe('MedlinePlus', () => {
  test('logo is visible on homepage', async ({ page }) => {
    await page.goto('https://medlineplus.gov/');

    const logo = page.getByAltText('MedlinePlus Trusted Health Information for You');
    await expect(logo).toBeVisible();

    await page.close();
  });
});
